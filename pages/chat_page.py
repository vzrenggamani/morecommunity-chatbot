import streamlit as st
import os


def show_chat_page(qa_chain):
    """Display the main chat interface"""

    # Add logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🏥 Rare Disease Helper Chatbot")

    st.markdown("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Mencari jawaban terbaik untuk Anda..."):
            # Get the response from the RAG chain
            response = qa_chain.invoke(prompt)
            answer = response["result"]

            # Display sources with document types for transparency
            sources = response["source_documents"]
            source_list = "\n\n**Sumber Informasi:**\n"

            # Get unique sources with types to avoid duplicates
            unique_sources = {}
            for doc in sources:
                source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                doc_type = doc.metadata.get("document_type", "general")

                # Map document types to Indonesian display names
                type_display = {
                    "medical_reference": "📚 Referensi Medis",
                    "user_story": "👥 Pengalaman Keluarga",
                    "community_resource": "🏘️ Sumber Komunitas",
                    "clinical_guideline": "⚕️ Pedoman Klinis",
                    "general": "📄 Umum",
                }

                display_type = type_display.get(doc_type, "📄 Umum")
                unique_sources[source_name] = display_type

            for source_name, source_type in sorted(unique_sources.items()):
                source_list += f"- {source_type}: {source_name}\n"

            # Create full response with sources only
            full_response = answer + source_list

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
