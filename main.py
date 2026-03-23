import streamlit as st
import uuid
from prompt_template import refine_prompt

def get_config():
    return {
        "search_endpoint": st.session_state.get("search_endpoint", ""),
        "search_key":      st.session_state.get("search_key", ""),
        "index_name":      st.session_state.get("index_name", "software-index"),
        "openai_endpoint": st.session_state.get("openai_endpoint", ""),
        "openai_key":      st.session_state.get("openai_key", ""),
        "openai_deploy":   st.session_state.get("openai_deploy", "text-embedding-3-small"),
    }

def config_is_set() -> bool:
    c = get_config()
    return all([c["search_endpoint"], c["search_key"], c["openai_endpoint"], c["openai_key"]])

def get_search_client():
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    c = get_config()
    return SearchClient(
        endpoint=c["search_endpoint"],
        index_name=c["index_name"],
        credential=AzureKeyCredential(c["search_key"]),
    )

def get_index_client():
    from azure.search.documents.indexes import SearchIndexClient
    from azure.core.credentials import AzureKeyCredential
    c = get_config()
    return SearchIndexClient(
        endpoint=c["search_endpoint"],
        credential=AzureKeyCredential(c["search_key"]),
    )

def get_openai_client():
    from openai import AzureOpenAI
    c = get_config()
    return AzureOpenAI(
        azure_endpoint=c["openai_endpoint"],
        api_key=c["openai_key"],
        api_version="2024-02-01",
    )

def get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model="gemma3:4b")

def create_index_if_not_exists():
    from azure.search.documents.indexes.models import (
        SearchIndex, SimpleField, SearchableField, SearchField,
        SearchFieldDataType, VectorSearch,
        HnswAlgorithmConfiguration, VectorSearchProfile,
    )
    c = get_config()
    index_client = get_index_client()
    existing = [i.name for i in index_client.list_indexes()]
    if c["index_name"] in existing:
        return

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="name", type=SearchFieldDataType.String),
        SearchableField(name="description", type=SearchFieldDataType.String),
        SearchableField(name="use_cases", type=SearchFieldDataType.String),
        SearchableField(name="category", type=SearchFieldDataType.String),
        SimpleField(
            name="tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")],
        profiles=[VectorSearchProfile(
            name="my-vector-profile",
            algorithm_configuration_name="my-hnsw",
        )],
    )
    index_client.create_index(
        SearchIndex(name=c["index_name"], fields=fields, vector_search=vector_search)
    )
    st.toast("Index created!", icon="✅")

def get_embedding(text: str) -> list[float]:
    c = get_config()
    response = get_openai_client().embeddings.create(
        input=text,
        model=c["openai_deploy"],
    )
    return response.data[0].embedding

def refine_query(raw_query: str) -> str | None:
    chain = refine_prompt | get_llm()
    result = chain.invoke({"query": raw_query}).content.strip()
    return None if result.upper() == "IRRELEVANT" else result

def search_software(query: str, top_k: int = 6) -> list:
    from azure.search.documents.models import VectorizedQuery
    query_vector = get_embedding(query)
    results = get_search_client().search(
        search_text=query,
        vector_queries=[VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )],
        select=["name", "description", "use_cases", "category", "tags"],
        top=top_k,
    )
    return list(results)

def add_software(name, description, use_cases, category, tags):
    embedding = get_embedding(f"{name}. {description}. Use cases: {use_cases}")
    get_search_client().upload_documents(documents=[{
        "id":          str(uuid.uuid4()),
        "name":        name,
        "description": description,
        "use_cases":   use_cases,
        "category":    category,
        "tags":        [t.strip() for t in tags.split(",")],
        "embedding":   embedding,
    }])

def delete_software(item_id: str):
    get_search_client().delete_documents(documents=[{"id": item_id}])

def list_all_software() -> list:
    return list(get_search_client().search(
        search_text="*",
        select=["id", "name", "category", "tags", "use_cases"],
        top=100,
    ))

with st.sidebar:
    st.header("API Configuration")
    st.text_input("Azure Search Endpoint",  key="search_endpoint", placeholder="https://xxx.search.windows.net")
    st.text_input("Azure Search Key",       key="search_key",      type="password")
    st.text_input("Index Name",             key="index_name",      placeholder="software-index")
    st.text_input("Azure OpenAI Endpoint",  key="openai_endpoint", placeholder="https://xxx.openai.azure.com")
    st.text_input("Azure OpenAI Key",       key="openai_key",      type="password")
    st.text_input("Embedding Deployment",   key="openai_deploy",   placeholder="text-embedding-3-small")

    if config_is_set():
        if st.button("Connect & Init Index"):
            with st.spinner("Connecting..."):
                try:
                    create_index_if_not_exists()
                    st.session_state["connected"] = True
                    st.success("Connected!")
                except Exception as e:
                    st.session_state["connected"] = False
                    st.error(f"Connection failed: {e}")
    else:
        st.info("Fill in all fields above to connect.")
        st.session_state["connected"] = False

st.title("Software Installation Agent")

if not st.session_state.get("connected"):
    st.warning("! Enter your API keys in the sidebar and click **Connect & Init Index** to get started.")
    st.stop()

tab1, tab2 = st.tabs(["Search", "Admin"])

with tab1:
    st.caption("Describe what you want to build — we'll find the right tools.")
    query = st.text_input("What are you trying to do?", placeholder="e.g. I want to build an AI app")

    if st.button("Find Tools") and query:
        with st.spinner("Understanding your query..."):
            refined = refine_query(query)

        if refined is None:
            st.warning("! That doesn't seem software-related. Try something like 'I want to build a chatbot'.")
        else:
            with st.spinner(f"Searching for: *{refined}*..."):
                results = search_software(refined)

            if not results:
                st.warning("No tools found. Try rephrasing.")
            else:
                st.caption(f"Searched for: *{refined}*")
                st.markdown(f"### Results for: *{query}*")
                cols = st.columns(2)
                for i, r in enumerate(results):
                    with cols[i % 2]:
                        st.markdown(f"#### {r['name']}")
                        st.caption(f"**Category:** {r['category']}")
                        st.write(r["description"])
                        st.write(f"**Use cases:** {r['use_cases']}")
                        st.markdown(f"`{'`  `'.join(r['tags'])}`")
                        st.divider()

with tab2:
    st.subheader("+ Add Software")

    with st.form("add_form"):
        name        = st.text_input("Name",       placeholder="e.g. LangChain")
        description = st.text_area("Description", placeholder="e.g. Framework for building LLM apps")
        use_cases   = st.text_area("Use Cases",   placeholder="e.g. AI apps, RAG pipelines, chatbots")
        category    = st.selectbox("Category", [
            "AI/ML", "Web Development", "DevOps", "Database",
            "Cloud", "Mobile", "Data Engineering", "Security", "Other",
        ])
        tags      = st.text_input("Tags (comma separated)", placeholder="e.g. llm, python, ai")
        submitted = st.form_submit_button("Add Software")

    if submitted:
        if not name or not description or not use_cases:
            st.error("Name, Description, and Use Cases are required.")
        else:
            with st.spinner("Embedding and uploading..."):
                add_software(name, description, use_cases, category, tags)
            st.success(f"'{name}' added successfully!")
            st.session_state["software_list"] = list_all_software()

    st.divider()
    st.subheader("All Software in Index")

    if st.button("Refresh List"):
        st.session_state["software_list"] = list_all_software()

    if "software_list" not in st.session_state:
        st.session_state["software_list"] = list_all_software()

    for item in st.session_state["software_list"]:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{item['name']}** — `{item['category']}`")
            st.caption(f"Tags: {', '.join(item['tags'])} | Use cases: {item['use_cases'][:80]}...")
        with col2:
            if st.button("Delete", key=item["id"]):
                delete_software(item["id"])
                st.success(f"Deleted '{item['name']}'")
                st.session_state["software_list"] = list_all_software()
                st.rerun()