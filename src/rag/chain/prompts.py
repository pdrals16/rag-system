from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEMPLATE = (
    "You are a helpful assistant. Answer the user's question using only the context "
    "provided below. If the context does not contain enough information to answer, "
    "say so clearly.\n\nContext:\n{context}"
)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])
