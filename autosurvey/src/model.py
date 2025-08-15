import os, time, openai, threading
from openai import AzureOpenAI
from .utils import tokenCounter
from typing import List, Optional

# Getting the AzureOpenAI variables
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
BASE_URL = os.getenv("AZURE_OPENAI_URL_BASE")
API_KEY = os.getenv("OPENAI_ORGANIZATION_KEY")

# Setting up the utilities for token length
token_counter = tokenCounter()


class APIModel:
    def __init__(
        self,
        model: str,
        api_key: str = API_KEY,
        azure_endpoint: str = BASE_URL,
        api_version: str = API_VERSION,
        token_limit: int = 128_000,
    ):
        self.__api_key = api_key
        self.__azure_endpoint = azure_endpoint
        self.__api_version = api_version
        self.model = model
        self.client = AzureOpenAI(
            api_version=self.__api_version,
            api_key=self.__api_key,
            azure_endpoint=self.__azure_endpoint,
        )
        self.token_limit = token_limit

    def __req(
        self,
        text: str,
        temperature: float,
        max_try: int = 5,
    ) -> Optional[str]:
        payload = {
            "messages": [{"role": "user", "content": text}],
            "model": self.model,
            "temperature": temperature,
        }

        # Trying the request with retries
        for attempt in range(max_try):

            try:
                response = self.client.chat.completions.create(**payload)
                return response.choices[0].message.content

            # Checking the content filter
            except openai.APIError as e:
                print(e)
                print(payload)

            except Exception as e:
                print(f"Request failed, attempt {attempt + 1} of {max_try}: {e}")
                time.sleep(2)  # Wait before retrying

        return None

    def chat(
        self,
        text: str,
        temperature: float = 1.0,
    ) -> str:
        response = self.__req(text, temperature=temperature, max_try=10)
        return response

    def __chat(
        self,
        text: str,
        temperature: float,
        res_l: List[str],
        idx: int,
    ) -> str:
        response = self.__req(
            text,
            temperature=temperature,
        )
        res_l[idx] = response
        return response

    def batch_chat(
        self,
        text_batch: List[str],
        temperature: float = 0.0,
    ) -> List[str]:
        max_threads = 15  # Limit the max concurrent threads using the model API
        res_l = ["No response"] * len(text_batch)
        thread_l = []
        for i, text in zip(range(len(text_batch)), text_batch):

            thread = threading.Thread(
                target=self.__chat,
                args=(text, temperature, res_l, i),
            )
            thread_l.append(thread)
            thread.start()
            while len(thread_l) >= max_threads:

                for t in thread_l:

                    if not t.is_alive():
                        thread_l.remove(t)

                time.sleep(0.3)  # Short delay to avoid busy-waiting

        for thread in thread_l:

            thread.join()

        return res_l
