import chainlit as cl


@cl.on_message
async def main(message: cl.Message):
    # TODO: Add logic to process the message
    # Send a message back to the user
    await cl.Message(content=f"Echo: {message.content}").send()
