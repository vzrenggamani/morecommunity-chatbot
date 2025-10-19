import streamlit as st
from datetime import datetime
from utils.llm_utils import get_llm_instance


def show_direct_llm_page():
    """Display direct LLM testing interface"""

    # Add logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🤖 Direct LLM Tester")

    st.markdown("---")

    st.markdown(
        """
    **Purpose:** This page calls the LLM directly without any RAG chain processing,
    allowing you to see the raw Google AI API response including `usage_metadata` and `response_metadata`.
    """
    )

    # Initialize session state for direct LLM responses
    if "direct_llm_responses" not in st.session_state:
        st.session_state.direct_llm_responses = []

    # Test query section
    st.header("🧪 Direct LLM Test")

    # Default test question
    default_question = "Apakah wajar saya marah ketika kerabat tidak mendukung saya ketika di masa sulit dengan anak saya yang autis?"

    test_query = st.text_area(
        "Enter your test question:",
        value=default_question,
        height=100,
        help="Enter any question to test the LLM directly and see complete metadata",
    )

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if st.button("🤖 Call LLM.invoke()", type="primary"):
            if test_query.strip():
                _call_llm_invoke(test_query)

    with col2:
        if st.button("📊 Call LLM.generate()"):
            if test_query.strip():
                _call_llm_generate(test_query)

    with col3:
        if st.button("🗑️ Clear"):
            st.session_state.direct_llm_responses = []
            st.success("✅ Cleared!")

    # Display responses
    if st.session_state.direct_llm_responses:
        st.header("📋 Direct LLM Response Analysis")

        # Response selector
        response_options = [
            f"{i+1}. {entry['timestamp']} - {entry['method']} - {entry['query'][:40]}{'...' if len(entry['query']) > 40 else ''}"
            for i, entry in enumerate(st.session_state.direct_llm_responses)
        ]

        selected_idx = st.selectbox(
            "Select response to analyze:",
            range(len(response_options)),
            format_func=lambda x: response_options[x],
        )

        if selected_idx is not None:
            selected_response = st.session_state.direct_llm_responses[selected_idx]
            _display_direct_llm_analysis(selected_response)

    else:
        st.info("💡 Click one of the buttons above to test direct LLM calls.")


def _call_llm_invoke(test_query):
    """Call LLM using invoke method"""
    with st.spinner("🤖 Calling llm.invoke()..."):
        try:
            llm = get_llm_instance()
            st.info(f"🔍 LLM instance type: {type(llm)}")

            # Call the LLM
            response = llm.invoke(test_query)
            st.info(f"✅ Response type: {type(response)}")

            # Process the response safely
            response_info = _process_response_safely(response, "invoke")

            # Store in session state
            response_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": "invoke",
                "query": test_query,
                "response_info": response_info,
                "raw_response": response,  # Keep for advanced analysis if needed
            }

            st.session_state.direct_llm_responses.insert(0, response_entry)
            if len(st.session_state.direct_llm_responses) > 10:
                st.session_state.direct_llm_responses = (
                    st.session_state.direct_llm_responses[:10]
                )

            st.success("✅ llm.invoke() completed successfully!")

        except Exception as e:
            st.error(f"❌ Error calling llm.invoke(): {e}")
            # Store error info
            error_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": "invoke (error)",
                "query": test_query,
                "response_info": {"error": str(e), "type": "Error"},
                "raw_response": None,
            }
            st.session_state.direct_llm_responses.insert(0, error_entry)


def _call_llm_generate(test_query):
    """Call LLM using generate method"""
    with st.spinner("📊 Calling llm.generate()..."):
        try:
            llm = get_llm_instance()
            st.info(f"🔍 LLM instance type: {type(llm)}")

            # Call the LLM with generate method
            response = llm.generate([[test_query]])
            st.info(f"✅ Response type: {type(response)}")

            # Process the response safely
            response_info = _process_response_safely(response, "generate")

            # Store in session state
            response_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": "generate",
                "query": test_query,
                "response_info": response_info,
                "raw_response": response,  # Keep for advanced analysis if needed
            }

            st.session_state.direct_llm_responses.insert(0, response_entry)
            if len(st.session_state.direct_llm_responses) > 10:
                st.session_state.direct_llm_responses = (
                    st.session_state.direct_llm_responses[:10]
                )

            st.success("✅ llm.generate() completed successfully!")

        except Exception as e:
            st.error(f"❌ Error calling llm.generate(): {e}")
            # Store error info
            error_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": "generate (error)",
                "query": test_query,
                "response_info": {"error": str(e), "type": "Error"},
                "raw_response": None,
            }
            st.session_state.direct_llm_responses.insert(0, error_entry)


def _process_response_safely(response, method):
    """Safely process LLM response to extract all available information"""

    response_info = {
        "method": method,
        "type": str(type(response)),
        "processing_steps": [],
    }

    # Try to get basic attributes
    for attr_name in ["content", "response_metadata", "usage_metadata"]:
        try:
            attr_value = getattr(response, attr_name, None)
            if attr_value is not None:
                response_info[attr_name] = str(attr_value)
                response_info["processing_steps"].append(
                    f"✅ Successfully got {attr_name}"
                )
            else:
                response_info[attr_name] = f"No {attr_name} attribute"
                response_info["processing_steps"].append(
                    f"⚠️ No {attr_name} attribute found"
                )
        except Exception as e:
            response_info[attr_name] = f"Error getting {attr_name}: {e}"
            response_info["processing_steps"].append(
                f"❌ Error getting {attr_name}: {e}"
            )

    # Try to get string representations
    try:
        response_info["full_str"] = str(response)
        response_info["processing_steps"].append(
            "✅ Successfully got string representation"
        )
    except Exception as e:
        response_info["full_str"] = f"Error getting string: {e}"
        response_info["processing_steps"].append(f"❌ Error getting string: {e}")

    try:
        response_info["full_repr"] = repr(response)
        response_info["processing_steps"].append("✅ Successfully got repr")
    except Exception as e:
        response_info["full_repr"] = f"Error getting repr: {e}"
        response_info["processing_steps"].append(f"❌ Error getting repr: {e}")

    # Try to get available attributes
    try:
        response_info["available_attributes"] = [
            attr for attr in dir(response) if not attr.startswith("_")
        ]
        response_info["processing_steps"].append(
            "✅ Successfully got available attributes"
        )
    except Exception as e:
        response_info["available_attributes"] = []
        response_info["processing_steps"].append(f"❌ Error getting attributes: {e}")

    return response_info


def _display_direct_llm_analysis(response_entry):
    """Display analysis of a direct LLM response"""

    st.subheader(f"📊 {response_entry['method'].upper()} Response Analysis")
    st.caption(f"Executed at: {response_entry['timestamp']}")

    # Show the original query
    with st.expander("📝 Original Query", expanded=False):
        st.text(response_entry["query"])

    response_info = response_entry["response_info"]

    # Show processing steps
    with st.expander("🔍 Processing Steps", expanded=False):
        for step in response_info.get("processing_steps", []):
            st.write(step)

    # Main response overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Method", response_info.get("method", "Unknown"))
    with col2:
        st.metric("Response Type", response_info.get("type", "Unknown"))
    with col3:
        attr_count = len(response_info.get("available_attributes", []))
        st.metric("Available Attributes", attr_count)

    # Show key information with highlights
    if "content" in response_info and "Error" not in response_info["content"]:
        with st.expander("💬 LLM Response Content", expanded=True):
            st.text_area("Content", response_info["content"], height=150, disabled=True)

    if (
        "usage_metadata" in response_info
        and "No usage_metadata" not in response_info["usage_metadata"]
    ):
        with st.expander("📊 Usage Metadata (TOKEN INFO!) 🎯", expanded=True):
            st.code(response_info["usage_metadata"], language="python")
    else:
        st.warning(
            "⚠️ No usage_metadata found - this might be expected depending on the LLM method used"
        )

    if (
        "response_metadata" in response_info
        and "No response_metadata" not in response_info["response_metadata"]
    ):
        with st.expander("📊 Response Metadata", expanded=True):
            st.code(response_info["response_metadata"], language="python")
    else:
        st.warning("⚠️ No response_metadata found")

    # Show available attributes
    if response_info.get("available_attributes"):
        with st.expander("🔧 Available Attributes", expanded=False):
            st.write("Available attributes on the response object:")
            for attr in response_info["available_attributes"]:
                st.write(f"- `{attr}`")

    # Show full representations
    with st.expander("📄 Full String Representation", expanded=False):
        st.code(response_info.get("full_str", "Not available"), language="python")

    with st.expander("📄 Full Repr Representation", expanded=False):
        st.code(response_info.get("full_repr", "Not available"), language="python")

    # Advanced analysis option
    if response_entry.get("raw_response") is not None:
        with st.expander("🔬 Advanced Raw Object Analysis", expanded=False):
            st.warning(
                "⚠️ This section tries to access the raw object directly - may cause errors"
            )
            if st.button(
                f"🔍 Analyze Raw {response_entry['method']} Object",
                key=f"analyze_{response_entry['timestamp']}",
            ):
                _analyze_raw_object(response_entry["raw_response"])


def _analyze_raw_object(raw_obj):
    """Advanced analysis of the raw object - with full error protection"""
    try:
        st.write(f"**Raw object type:** `{type(raw_obj)}`")

        # Try to access each attribute individually
        for attr_name in ["content", "response_metadata", "usage_metadata"]:
            try:
                attr_value = getattr(raw_obj, attr_name, "ATTRIBUTE_NOT_FOUND")
                if attr_value != "ATTRIBUTE_NOT_FOUND":
                    st.write(f"**{attr_name}:**")
                    if hasattr(attr_value, "__dict__") or isinstance(attr_value, dict):
                        st.json(
                            attr_value
                            if isinstance(attr_value, dict)
                            else attr_value.__dict__
                        )
                    else:
                        st.code(str(attr_value))
                else:
                    st.write(f"**{attr_name}:** Not found")
            except Exception as e:
                st.error(f"Error accessing {attr_name}: {e}")

    except Exception as e:
        st.error(f"Error in advanced analysis: {e}")
