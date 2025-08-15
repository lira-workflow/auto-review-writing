import os, json, time, random
from argparse import Namespace
from openai import AzureOpenAI, OpenAIError
from utils.constants import ASSISTANT_NAME
from baseline_and_mass.utils.text_utils import check_length, check_token_count


# NOTE: Please first set the below OpenAI variables as described in the README
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
base_url = os.getenv("AZURE_OPENAI_URL_BASE")
api_key = os.getenv("OPENAI_ORGANIZATION_KEY")

# Setting up the OpenAI client
client = AzureOpenAI(
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=base_url,
)


def retry_with_backoff(
    func, *args, max_retries=5, base_delay=1, max_delay=60, **kwargs
):
    """
    Retries a function on 429 errors with exponential backoff.
    """

    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)

        except OpenAIError as e:
            if "429" in str(e):
                delay = min(base_delay * 2**retries + random.uniform(0, 1), max_delay)
                print(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                retries += 1

            else:
                raise

    raise Exception("Max retries reached. Could not complete the request.")


def prompt_gpt(
    prompt: str,
    input_file: str,
    base_prompt: str,
    model: str = "gpt-4o-2024-05-13",
    temperature: float = 0.0,
) -> str:
    """
    Prompts a specified ChatGPT model from AzureOpenAI.
    Automatically splits up the prompt into smaller pieces based on its length.

    Parameters:
    prompt - the prompt to send to ChatGPT.
    input_file - the file to upload (only used if the resulting prompt is too long).
    base_prompt - the original prompt, unformatted (only used if the resulting prompt is too long).
    model - the OpenAI chat model to use.
    temperature - the output temperature (higher value results in more response variance).

    Returns:
    the model response as text.
    """

    # Attach as a file if the prompt is too long
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if check_length(prompt) or check_token_count(prompt):
        return prompt_gpt_file(
            input_file, base_prompt, model=model, temperature=temperature
        )

    messages.append({"role": "user", "content": prompt})
    return retry_with_backoff(
        lambda: client.chat.completions.create(
            model=model, messages=messages, stream=False, temperature=temperature
        )
        .choices[0]
        .message.content
    )


def prompt_gpt_file(input_file: str, base_prompt: str, model: str, temperature: float):
    """
    Prompts the LLM with an attachment uploaded.

    Parameters:
    input_file - the file to upload.
    base_prompt - the original prompt, unformatted.
    model - the OpenAI chat model to use.
    temperature - the output temperature (higher value results in more response variance).

    Returns:
    the model response if successful.
    """

    try:
        # Setup the assistant (only if not already present)
        assistants = retry_with_backoff(client.beta.assistants.list)
        assistant = next((a for a in assistants if a.name == ASSISTANT_NAME), None)

        if not assistant:
            assistant = retry_with_backoff(
                client.beta.assistants.create,
                name=ASSISTANT_NAME,
                instructions="You are a helpful assistant.",
                tools=[{"type": "file_search"}],
                model=model,
                temperature=temperature,
            )

        # Setup the messages
        message_file = retry_with_backoff(
            lambda: client.files.create(
                file=open(input_file, "rb"), purpose="assistants"
            )
        )
        messages = [
            {
                "role": "user",
                "content": base_prompt.format(
                    content="[The contents are in the attached file]"
                ),
                "attachments": [
                    {"file_id": message_file.id, "tools": [{"type": "file_search"}]}
                ],
            }
        ]

        # Setup the other components
        thread = retry_with_backoff(client.beta.threads.create, messages=messages)
        run = retry_with_backoff(
            client.beta.threads.runs.create,
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        # Execute the run
        while run.status not in ["completed", "cancelled", "expired", "failed"]:
            time.sleep(10)
            run = retry_with_backoff(
                client.beta.threads.runs.retrieve, thread_id=thread.id, run_id=run.id
            )

        messages = retry_with_backoff(
            client.beta.threads.messages.list, thread_id=thread.id
        )

        # Delete the files (as otherwise they are retained on the AzureOpenAI servers)
        client.files.delete(message_file.id)
        client.beta.threads.delete(thread_id=thread.id)

        completion = json.loads(messages.model_dump_json())["data"][0]["content"][0][
            "text"
        ]["value"]

        return completion

    except Exception as e:
        print(f"Error processing file `{input_file}`: {e}")
        return None


def prompt_llm(prompt: str, input_file: str, base_prompt: str, args: Namespace) -> str:
    """
    Prompts an LLM based on the user choice.

    Parameters:
    prompt - the prompt to send to the LLM.
    input_file - the file to upload (only used if the resulting prompt is too long).
    base_prompt - the original prompt, unformatted (only used if the resulting prompt is too long).
    args - a Namespace containing the following variables which get read:

    input_folder - the directory containing the sample `_references.txt` files.
    model - the model to use.
    temperature - the temperature for model prompting.
    """

    if "gpt" in args.model.lower():
        completion = prompt_gpt(
            prompt,
            input_file,
            base_prompt,
            model=args.model,
            temperature=args.temperature,
        )

    else:
        raise ValueError(f"`{args.model}` is currently not supported!")

    return completion
