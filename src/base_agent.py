import gc, os, re, tiktoken
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from typing import Dict, List, Union, Optional
from langchain.schema import AIMessage, HumanMessage
from src.utils.file_utils import load_json, save_to_json
from src.utils.constants import (
    LLM_SEED,
    API_KEY,
    BASE_URL,
    TEMP_DIR,
    API_VERSION,
    MODEL_VER,
    MAX_ITEMS,
    TEMPERATURE,
    CONTEXT_SIZE,
    BEGIN_PROMPT,
    SETTING_NAME,
    ENCODING_NAME,
)


# The Base Agent class that all specific agents inherit from
class BaseAgent:
    def __init__(
        self,
        name: str,
        review_id: str,
        model_name: str = MODEL_VER,
        context_size: int = CONTEXT_SIZE,  # The size in tokens
        temperature: float = TEMPERATURE,
        max_memory_items: int = MAX_ITEMS,
        system_prompt: str = "You are a helpful assistant.",
        dataset: str = "srg",
        temp_dir: str = TEMP_DIR,
        encoding_name: str = ENCODING_NAME,
        setting_name: str = SETTING_NAME,
        overwrite_response: bool = False,
        seed: int = LLM_SEED,
    ):
        self.name = name
        self.review_id = review_id

        # Determine the model based on the name provided
        if "gpt" in model_name.lower():
            self.model = AzureChatOpenAI(
                model_name=model_name,
                api_version=API_VERSION,
                azure_endpoint=BASE_URL,
                api_key=API_KEY,
                temperature=temperature,
            )

        else:
            # Ensure the model is already installed beforehand
            self.model = ChatOllama(
                model=model_name,
                temperature=temperature,
                num_ctx=context_size,
                num_predict=3_000,
                seed=seed,
                disable_streaming=True,
            )

        self.context_size = context_size
        self.max_memory_items = max_memory_items
        self.system_prompt = system_prompt
        self.dataset = dataset
        self.overwrite_response = overwrite_response

        # To ensure that certain components (mainly the paper analyses do not get redone)
        if setting_name.lower != "general":
            self.temp_dir = os.path.join(temp_dir, dataset, setting_name, review_id)

        else:
            self.temp_dir = os.path.join(temp_dir, dataset, review_id)

        self.encoding_name = encoding_name
        self.setting_name = setting_name

        # Make the temporary directory
        os.makedirs(self.temp_dir, exist_ok=True)

    # For counting the human message lengths
    def count_tokens(self, string: str) -> int:
        """
        Calculates the number of tokens a prompt uses.

        Parameters:
        string - the string to count the length of.

        Returns:
        the length of the string in tokens.
        """

        encoding = tiktoken.encoding_for_model(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def filter_messages(self, messages: List) -> List:
        """
        Filters out messages based on the following criteria:

        Parameters:
        messages - the messages for the current discussion (as separate discussion lists are made for ease).

        Returns:
        a list of messages filtered based on the aforementioned criteria.
        """

        # First remove all system messages
        message_list = [
            message
            for message in messages
            if message.content != BEGIN_PROMPT and "UPDATE: " not in message.content
        ]

        # Then check if there are too many items in the list
        total = len(message_list)
        if total > self.max_memory_items:
            message_list = message_list[total - self.max_memory_items : total]

        # Then check if the token length is too long for the whole conversation
        n_tokens = sum(
            [
                self.count_tokens(message.content)
                for message in message_list
                if isinstance(message, AIMessage) or isinstance(message, HumanMessage)
            ]
        )

        # If yes, filter the messages (take the latest first)
        if n_tokens > self.context_size:
            print("Context too long. Filtering...\n")
            count = 0
            message_list_final = []
            message_list.reverse()
            for message in message_list:

                count += self.count_tokens(message.content)
                if count >= self.context_size:
                    break

                message_list_final.append(message)

            # Re-reverse the messages
            message_list_final.reverse()

        else:
            message_list_final = message_list

        return message_list_final

    def call_model(
        self,
        prompt: Optional[str],
        messages: Optional[List] = None,
        response_only: bool = False,
        to_update: Optional[str] = None,
        custom_update_val: Optional[Union[int, str]] = None,
        folder_name: Optional[str] = None,
        file_name: str = "temp.json",
        do_save: bool = True,
        include_human_prompt: bool = False,
        as_dict: bool = False,
    ) -> Union[Dict, str]:
        """
        Creates a call to the LLM and saves the response.
        The saved response is then used later to reduce costs and allow for more output checking.

        Parameters:
        prompt - the prompt to feed to the model.
        messages - a message list to use if desired (mainly for the agent discussion to create different discussions).
        response_only - whether to only return the response or the full conversation history.
        to_update - the value in the state to update using either the response or another value.
        custom_update_val - the value to add to the state (to the key defined by `to_update`) if provided,
        else it updates the state using the model response.
        folder_name - the name of the subfolder to save in if provided.
        file_name - the filename to use for saving the response.
        do_save - whether to save the LLM output.
        include_human_prompt - whether to include the human prompt or not.
        as_dict - whether to output a dictionary or a string.

        Returns:
        a dictionary or string containing either the full update or only the model response, alongside other values
        that want to updated.
        """

        # Check if the `file_name` ends with `.json`
        assert file_name.endswith(
            ".json"
        ), "Please make sure all save files end with `json`!"

        # Also fix if any slashes are present
        file_name = re.sub(r"(\\|/)", "", file_name)
        human_msg = HumanMessage(content=f"{self.system_prompt}\n\n{prompt}")

        # Setup the target folder if provided and load the response if available
        # (and if required)
        if folder_name is not None and do_save:
            new_dir = os.path.join(self.temp_dir, folder_name)
            os.makedirs(new_dir, exist_ok=True)
            target_file = os.path.join(new_dir, file_name)

        else:
            target_file = os.path.join(self.temp_dir, file_name)

        # Prompt the LLM
        if os.path.isfile(target_file) and not self.overwrite_response:
            data = load_json(target_file)
            response = AIMessage(**data)

        else:
            # Otherwise prompt the model
            messages_invoke = (
                self.filter_messages(messages + [human_msg])
                if messages is not None
                else [human_msg]
            )

            response = self.model.invoke(messages_invoke)
            data = response.model_dump(mode="json")
            if do_save:
                save_to_json(data, target_file)

        # Adding values to update the state with via the command
        new_messages = [human_msg, response] if include_human_prompt else [response]
        message_list = new_messages if response_only else messages + new_messages
        update = {"messages": message_list}
        if to_update:
            update[to_update] = response if not custom_update_val else custom_update_val

        # Memory Leak Prevention
        del data, message_list
        gc.collect()

        # Returning the final output
        if as_dict:
            return update

        else:
            return update["messages"]
