import streamlit as st
import tiktoken
from datetime import datetime


def count_tokens(text, model_name="gpt-3.5-turbo"):
    """Count tokens in text using tiktoken"""
    try:
        # Use a general tokenizer for Gemini models
        encoding = tiktoken.get_encoding(
            "cl100k_base"
        )  # This is used by GPT-4 and similar models
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback: rough estimation (1 token ≈ 4 characters for most languages)
        return len(text) // 4


def initialize_token_tracking():
    """Initialize token usage tracking in session state"""
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "conversation_count": 0,
            "session_history": [],
        }


def add_token_usage(input_tokens, output_tokens, query_text, response_text):
    """Add token usage to session tracking"""
    if "token_usage" not in st.session_state:
        initialize_token_tracking()

    st.session_state.token_usage["total_input_tokens"] += input_tokens
    st.session_state.token_usage["total_output_tokens"] += output_tokens
    st.session_state.token_usage["conversation_count"] += 1

    # Add to history (keep last 10 conversations)
    conversation_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "query_preview": (
            query_text[:50] + "..." if len(query_text) > 50 else query_text
        ),
        "response_preview": (
            response_text[:50] + "..." if len(response_text) > 50 else response_text
        ),
    }

    st.session_state.token_usage["session_history"].append(conversation_entry)

    # Keep only last 10 conversations
    if len(st.session_state.token_usage["session_history"]) > 10:
        st.session_state.token_usage["session_history"] = st.session_state.token_usage[
            "session_history"
        ][-10:]


def format_token_info(input_tokens, output_tokens, context_text, num_retrieved_docs):
    """Format token usage information for display"""
    total_tokens = input_tokens + output_tokens

    # Estimate cost (using approximate pricing for Gemini Pro)
    input_cost = (input_tokens / 1000) * 0.00025  # $0.00025 per 1K input tokens
    output_cost = (output_tokens / 1000) * 0.0005  # $0.0005 per 1K output tokens
    total_cost = input_cost + output_cost

    context_tokens = count_tokens(context_text) if context_text else 0

    token_info = f"""

    **📊 Token Usage:**
    - Input: {input_tokens:,} tokens
    - Output: {output_tokens:,} tokens
    - Total: {total_tokens:,} tokens
    - Context: {context_tokens:,} tokens (from {num_retrieved_docs} docs)
    - Estimated cost: ${total_cost:.6f}

    **📈 Session Stats:**
    - Total conversations: {st.session_state.token_usage['conversation_count']}
    - Total input tokens: {st.session_state.token_usage['total_input_tokens']:,}
    - Total output tokens: {st.session_state.token_usage['total_output_tokens']:,}
    - Session cost: ${(st.session_state.token_usage['total_input_tokens'] * 0.00025 + st.session_state.token_usage['total_output_tokens'] * 0.0005) / 1000:.6f}
    """
    return token_info


def get_optimized_context(relevant_docs, max_tokens=500):
    """Get context text optimized for token usage"""
    context_parts = []
    current_tokens = 0

    for i, doc in enumerate(relevant_docs):
        doc_content = doc.page_content
        doc_tokens = count_tokens(doc_content)

        # If adding this document would exceed limit, truncate it
        if current_tokens + doc_tokens > max_tokens:
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 50:  # Only add if we have meaningful space left
                # Truncate the document to fit
                words = doc_content.split()
                truncated_content = ""
                for word in words:
                    test_content = (
                        truncated_content + " " + word if truncated_content else word
                    )
                    if count_tokens(test_content) <= remaining_tokens:
                        truncated_content = test_content
                    else:
                        break
                if truncated_content:
                    context_parts.append(truncated_content + "...")
            break
        else:
            context_parts.append(doc_content)
            current_tokens += doc_tokens

    return "\n\n".join(context_parts), len(context_parts)
