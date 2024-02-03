from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen


# instaed of manually calling everytime like print(boxen("TEST HERE!!", title="Human", color="yellow")), we created a funciton.
def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))


class ChatModelStartHandler(BaseCallbackHandler):
    # serialized:- less used. its type is Dict[str, Any]
    # messages:- MOst important. Its type is List[List[BaseMessage]]

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n ============ Sending Messages ============\n\n")

        for message in messages[0]:
            print("type of the message is: ", type(message))

            # to display system message
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")
            # to display human message
            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")
            # if we(langchain-app) are getting back a function to be executed by us from LLM.
            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"]
                boxen_print(
                    f"Running tool {call['name']} with args {call['arguments']}",
                    title=message.type,
                    color="cyan",
                )
            # to display AI message
            elif message.type == "ai":
                boxen_print(message.content, title=message.type, color="blue")
            # if we are sending back the result of a function call to LLM from langchain.
            elif message.type == "function":
                boxen_print(
                    message.content,
                    title=f"{message.type} executed by us and the result sent back to LLM",
                    color="purple",
                )

            else:
                boxen_print(message.content, title=message.type, color="red")
