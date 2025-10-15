import streamlit as st
import os
from utils.token_tracking import (
    initialize_token_tracking,
    add_token_usage,
    count_tokens,
    format_token_info,
)


def show_chat_page(qa_chain):
    """Display the main chat interface"""
    # Initialize token tracking
    initialize_token_tracking()

    # Add logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🏥 Rare Disease Helper Chatbot")

    st.markdown("---")

    # Display token usage summary in sidebar
    with st.sidebar:
        if st.session_state.get("token_usage"):
            st.markdown("### 📊 Token Usage (This Session)")
            usage = st.session_state.token_usage
            st.metric("Total Conversations", usage["conversation_count"])
            st.metric("Total Input Tokens", f"{usage['total_input_tokens']:,}")
            st.metric("Total Output Tokens", f"{usage['total_output_tokens']:,}")
            st.metric(
                "Total Tokens",
                f"{usage['total_input_tokens'] + usage['total_output_tokens']:,}",
            )

            # Rough cost estimation
            total_cost = (usage["total_input_tokens"] * 0.000001) + (
                usage["total_output_tokens"] * 0.000002
            )
            st.metric("Estimated Cost", f"${total_cost:.6f}")

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
            # Calculate input tokens (context + prompt)
            retriever = qa_chain.retriever
            relevant_docs = retriever.get_relevant_documents(prompt)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Debug: Show context statistics
            context_length = len(context_text)
            num_retrieved_docs = len(relevant_docs)

            # Calculate tokens for the full prompt that will be sent to LLM
            full_prompt_text = f"""
            Anda adalah dokter AI yang ahli dalam bidang penyakit langka dan kesehatan. Anda memiliki akses ke berbagai jenis pengetahuan medis:

            SUMBER PENGETAHUAN ANDA:
            {context_text}

            Ketika memberikan jawaban, sesuaikan pendekatan berdasarkan jenis informasi:
            - Informasi medis/klinis: Gunakan dengan otoritas profesional dan presisi klinis
            - Pengalaman pasien: Gunakan untuk memberikan empati dan pemahaman mendalam tentang pengalaman keluarga
            - Sumber komunitas: Gunakan untuk memberikan dukungan dan informasi praktis tentang resources

            PERTANYAAN PASIEN: {prompt}

            PENTING - JANGAN LAKUKAN:
            - JANGAN katakan "berdasarkan cerita Anda", "dari informasi yang Anda berikan"
            - JANGAN asumsikan pasien telah menceritakan detail yang ada di pengetahuan Anda
            - JANGAN rujuk ke sumber atau dokumen secara eksplisit

            LAKUKAN:
            - Jawab langsung dan profesional seperti dokter berpengalaman
            - Integrasikan semua jenis pengetahuan secara natural
            - Berikan empati berdasarkan pengalaman keluarga yang Anda ketahui
            - Sertakan informasi praktis dan dukungan komunitas jika relevan
            - Selalu sarankan konsultasi medis profesional untuk diagnosis dan pengobatan

            JAWABAN DOKTER:"""

            input_tokens = count_tokens(full_prompt_text)

            # Get the response from the RAG chain
            response = qa_chain.invoke(prompt)
            answer = response["result"]

            # Calculate output tokens
            output_tokens = count_tokens(answer)

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

            # Add token usage information
            token_info = format_token_info(
                input_tokens, output_tokens, context_text, num_retrieved_docs
            )

            # Create full response with token info
            full_response = answer + source_list + token_info

            # Track token usage
            add_token_usage(input_tokens, output_tokens, prompt, answer)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
