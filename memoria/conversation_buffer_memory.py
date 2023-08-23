from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


class userQuestion(BaseModel):
    question: str
    conversation_id: str

llm = OpenAI(openai_api_key="...",temperature=0)
memory = ConversationBufferMemory()
memories = {}

CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Eres un experto en hablar como un ruso que apenas esta aprendiendo español."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

app = FastAPI()

@app.post("/conversation/buffer_memory")
async def chat_memory(data: userQuestion):
    memory = memories.get(data.conversation_id)
    
    if memory is None:
        memory = ConversationBufferMemory(return_messages=True)
        memories[data.conversation_id] = memory

   
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory
    )

    ia_response = conversation.predict(input=data.question)
    memory.save_context({"input": data.question}, {"output": ia_response})
    return ia_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)