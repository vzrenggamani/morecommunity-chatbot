import streamlit as st
from datetime import datetime
from utils.llm_utils import get_llm_instance


def show_raw_response_page(qa_chain):
    """Display raw response inspection interface"""

    # Add logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🔍 Raw Response Inspector")

    st.markdown("---")

    st.markdown(
        """
    **Purpose:** This page shows the complete raw response from the RAG chain invoke method,
    including all metadata and internal structures that are normally hidden from users.
    """
    )

    # Initialize session state for raw responses
    if "raw_responses" not in st.session_state:
        st.session_state.raw_responses = []

    # Test query section
    st.header("🧪 Test Query")

    # Default test question
    default_question = "Apakah wajar saya marah ketika kerabat tidak mendukung saya ketika di masa sulit dengan anak saya yang autis?"

    test_query = st.text_area(
        "Enter your test question:",
        value=default_question,
        height=100,
        help="Enter any question to see the complete raw response structure",
    )

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        if st.button("🔍 Execute Query", type="primary"):
            if test_query.strip():
                with st.spinner("Executing query and capturing raw response..."):
                    try:
                        # Execute the RAG chain query
                        raw_response = qa_chain.invoke(test_query)

                        # Also get the direct LLM response for comparison
                        try:
                            llm = get_llm_instance()
                            direct_llm_response = llm.invoke(test_query)

                            # Try to get the full response with metadata
                            llm_response_with_metadata = llm.generate([[test_query]])
                        except Exception as llm_error:
                            # Create a safe error object instead of a string
                            direct_llm_response = {
                                "type": "Error",
                                "error": f"Error getting direct LLM response: {llm_error}",
                                "content": "Error - no content available",
                                "response_metadata": "Error - no metadata available",
                                "usage_metadata": "Error - no usage metadata available"
                            }
                            llm_response_with_metadata = {
                                "type": "Error",
                                "error": f"Error getting LLM metadata: {llm_error}"
                            }

                        # Store the response with timestamp
                        response_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "query": test_query,
                            "rag_response": raw_response,
                            "direct_llm_response": direct_llm_response,
                            "llm_response_with_metadata": llm_response_with_metadata,
                        }

                        # Add to session state (keep last 5 responses)
                        st.session_state.raw_responses.insert(0, response_entry)
                        if len(st.session_state.raw_responses) > 5:
                            st.session_state.raw_responses = (
                                st.session_state.raw_responses[:5]
                            )

                        st.success("✅ Query executed successfully!")

                    except Exception as e:
                        st.error(f"❌ Error executing query: {str(e)}")

    with col2:
        if st.button("🔍 Direct LLM Only"):
            if test_query.strip():
                with st.spinner("Calling LLM directly to capture metadata..."):
                    try:
                        llm = get_llm_instance()

                        # Multiple ways to call the LLM to capture different metadata
                        st.info("🔍 Calling llm.invoke()...")
                        direct_response = llm.invoke(test_query)
                        st.info(f"✅ llm.invoke() returned: {type(direct_response)}")

                        # st.info("🔍 Calling llm.generate()...")
                        # generate_response = llm.generate([[test_query]])
                        # st.info(f"✅ llm.generate() returned: {type(generate_response)}")

                        # Convert to simple representations to avoid attribute errors
                        st.info("🔍 Processing direct_response...")
                        try:
                            # Safe attribute access with individual try-catch
                            content_val = "No content attribute"
                            try:
                                content_val = str(getattr(direct_response, 'content', 'No content attribute'))
                                st.info("✅ Successfully got content attribute")
                            except Exception as e:
                                st.error(f"❌ Error getting content: {e}")

                            response_metadata_val = "No response_metadata"
                            try:
                                response_metadata_val = str(getattr(direct_response, 'response_metadata', 'No response_metadata'))
                                st.info("✅ Successfully got response_metadata attribute")
                            except Exception as e:
                                st.error(f"❌ Error getting response_metadata: {e}")

                            usage_metadata_val = "No usage_metadata"
                            try:
                                usage_metadata_val = str(getattr(direct_response, 'usage_metadata', 'No usage_metadata'))
                                st.info("✅ Successfully got usage_metadata attribute")
                            except Exception as e:
                                st.error(f"❌ Error getting usage_metadata: {e}")

                            full_str_val = "Error converting to string"
                            try:
                                full_str_val = str(direct_response)
                                st.info("✅ Successfully converted to string")
                            except Exception as e:
                                st.error(f"❌ Error converting to string: {e}")

                            full_repr_val = "Error getting repr"
                            try:
                                full_repr_val = repr(direct_response)
                                st.info("✅ Successfully got repr")
                            except Exception as e:
                                st.error(f"❌ Error getting repr: {e}")

                            dir_attrs = []
                            try:
                                dir_attrs = [attr for attr in dir(direct_response) if not attr.startswith('_')]
                                st.info("✅ Successfully got dir attributes")
                            except Exception as e:
                                st.error(f"❌ Error getting dir: {e}")

                            direct_response_info = {
                                "type": str(type(direct_response)),
                                "content": content_val,
                                "response_metadata": response_metadata_val,
                                "usage_metadata": usage_metadata_val,
                                "full_object_str": full_str_val,
                                "full_object_repr": full_repr_val,
                                "dir_attributes": dir_attrs
                            }
                            st.info("✅ direct_response processed successfully")
                        except Exception as e:
                            st.error(f"❌ Error processing direct_response: {e}")
                            direct_response_info = {
                                "type": str(type(direct_response)),
                                "error": str(e),
                                "full_object_str": "Error getting string representation",
                            }

                        st.info("🔍 Processing generate_response...")
                        try:
                            generate_response_info = {
                                "type": str(type(generate_response)),
                                "full_object_str": str(generate_response),
                                "full_object_repr": repr(generate_response),
                                "dir_attributes": [attr for attr in dir(generate_response) if not attr.startswith('_')]
                            }
                            st.info("✅ generate_response processed successfully")
                        except Exception as e:
                            st.error(f"❌ Error processing generate_response: {e}")
                            generate_response_info = {
                                "type": str(type(generate_response)),
                                "error": str(e),
                                "full_object_str": str(generate_response),
                            }

                        st.info("🔍 Creating response entry...")
                        # Store the response with timestamp (NO ORIGINAL OBJECTS!)
                        response_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "query": test_query,
                            "rag_response": None,
                            "direct_llm_response": direct_response_info,
                            "llm_response_with_metadata": generate_response_info,
                        }

                        st.info("🔍 Adding to session state...")
                        # Add to session state (keep last 5 responses)
                        st.session_state.raw_responses.insert(0, response_entry)
                        if len(st.session_state.raw_responses) > 5:
                            st.session_state.raw_responses = (
                                st.session_state.raw_responses[:5]
                            )
                        st.info("✅ Successfully added to session state!")

                        st.success("✅ Direct LLM call executed successfully!")

                    except Exception as e:
                        st.error(f"❌ Error executing direct LLM call: {str(e)}")

    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.raw_responses = []
            st.success("✅ Response history cleared!")

    # Display raw responses
    if st.session_state.raw_responses:
        st.header("📋 Raw Response Analysis")

        # Response selector
        response_options = [
            f"{i+1}. {entry['timestamp']} - {entry['query'][:50]}{'...' if len(entry['query']) > 50 else ''}"
            for i, entry in enumerate(st.session_state.raw_responses)
        ]

        selected_idx = st.selectbox(
            "Select response to analyze:",
            range(len(response_options)),
            format_func=lambda x: response_options[x],
        )

        if selected_idx is not None:
            selected_response = st.session_state.raw_responses[selected_idx]
            _display_comprehensive_analysis(selected_response)

    else:
        st.info("💡 Execute a query above to see the raw response structure.")


def _display_comprehensive_analysis(response_entry):
    """Display comprehensive analysis of RAG and direct LLM responses"""

    st.subheader("📊 Comprehensive Response Analysis")
    st.caption(f"Executed at: {response_entry['timestamp']}")

    # Show the original query
    with st.expander("📝 Original Query", expanded=False):
        st.text(response_entry["query"])

    # Create tabs for different response types
    tabs = []
    if response_entry.get("rag_response"):
        tabs.append("🔗 RAG Chain Response")
    if response_entry.get("direct_llm_response"):
        tabs.append("🤖 Direct LLM Response")
    if response_entry.get("llm_response_with_metadata"):
        tabs.append("📊 LLM with Full Metadata")

    if not tabs:
        st.error("No response data available")
        return

    tab_objects = st.tabs(tabs)

    tab_idx = 0

    # RAG Chain Response Tab
    if response_entry.get("rag_response"):
        with tab_objects[tab_idx]:
            st.markdown("**This is the response from the complete RAG chain (qa_chain.invoke)**")
            _display_single_response_analysis("RAG Chain", response_entry["rag_response"])
        tab_idx += 1

    # Direct LLM Response Tab
    if response_entry.get("direct_llm_response"):
        with tab_objects[tab_idx]:
            st.markdown("**This is the direct LLM response (llm.invoke)**")
            _display_single_response_analysis("Direct LLM", response_entry["direct_llm_response"])
        tab_idx += 1

    # LLM with Full Metadata Tab
    if response_entry.get("llm_response_with_metadata"):
        with tab_objects[tab_idx]:
            st.markdown("**This is the LLM response with full metadata (llm.generate)**")
            st.info("🎯 This should contain usage_metadata and response_metadata!")
            _display_single_response_analysis("LLM Full Metadata", response_entry["llm_response_with_metadata"])


def _display_single_response_analysis(response_name, response_data):
    """Display analysis of a single response"""

    if response_data is None:
        st.warning(f"No {response_name} data available")
        return

    # Response structure overview
    st.subheader(f"🏗️ {response_name} Structure Overview")

    # Check if this is our simplified string representation
    if isinstance(response_data, dict) and "type" in response_data and "full_object_str" in response_data:
        _display_simplified_response(response_data)
    else:
        # Original handling for other response types
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Response Type", type(response_data).__name__)

        with col2:
            if hasattr(response_data, "__len__"):
                st.metric("Response Size", len(str(response_data)))
            else:
                st.metric("Response Size", "N/A")

        with col3:
            if isinstance(response_data, dict):
                st.metric("Top-level Keys", len(response_data.keys()))
            elif hasattr(response_data, "__dict__"):
                st.metric("Attributes", len([attr for attr in dir(response_data) if not attr.startswith("_")]))
            else:
                st.metric("Top-level Keys", "N/A")

        # Main content analysis
        if isinstance(response_data, dict):
            _display_dict_response(response_data)
        elif hasattr(response_data, "__dict__"):
            _display_object_response(response_data)
        else:
            _display_simple_response(response_data)

        # Raw JSON view
        st.subheader("🗂️ Complete Raw Response (JSON)")
        with st.expander("Show complete raw response as JSON", expanded=False):
            try:
                # Convert to JSON-serializable format
                json_response = _make_json_serializable(response_data)
                st.json(json_response)
            except Exception as e:
                st.error(f"Cannot convert to JSON: {e}")
                st.code(str(response_data), language="python")


def _display_simplified_response(response_info):
    """Display simplified string-based response info"""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Response Type", response_info["type"])
    with col2:
        st.metric("Has Content", "Yes" if "content" in response_info else "No")
    with col3:
        st.metric("Attributes Count", len(response_info.get("dir_attributes", [])))

    # Show key information
    if "content" in response_info:
        with st.expander("💬 LLM Response Content", expanded=True):
            st.text_area("Content", response_info["content"], height=150, disabled=True)

    if "response_metadata" in response_info and response_info["response_metadata"] != "No response_metadata":
        with st.expander("📊 Response Metadata", expanded=True):
            st.code(response_info["response_metadata"], language="python")

    if "usage_metadata" in response_info and response_info["usage_metadata"] != "No usage_metadata":
        with st.expander("📊 Usage Metadata (TOKEN INFO!)", expanded=True):
            st.code(response_info["usage_metadata"], language="python")

    # Show all available attributes
    if "dir_attributes" in response_info:
        with st.expander("🔧 Available Attributes", expanded=False):
            st.write("Available attributes on the original object:")
            for attr in response_info["dir_attributes"]:
                st.write(f"- `{attr}`")

    # Show string representations
    with st.expander("📄 Full Object String Representation", expanded=False):
        st.code(response_info["full_object_str"], language="python")

    with st.expander("📄 Full Object Repr", expanded=False):
        st.code(response_info["full_object_repr"], language="python")
def _display_dict_response(response_data):
    """Display analysis for dictionary responses"""

    # Show all top-level keys
    st.subheader("🔑 All Top-Level Keys")
    keys_info = []
    for key in response_data.keys():
        value = response_data[key]
        keys_info.append(
            {
                "Key": key,
                "Type": type(value).__name__,
                "Length": len(value) if hasattr(value, "__len__") else "N/A",
                "Preview": (
                    str(value)[:100] + "..."
                    if len(str(value)) > 100
                    else str(value)
                ),
            }
        )

    st.dataframe(keys_info, use_container_width=True)

    # Special handling for known important keys
    if "result" in response_data:
        with st.expander("🎯 Main Result (User-Facing Content)", expanded=True):
            st.markdown("**This is what users normally see:**")
            st.markdown(response_data["result"])

    if "usage_metadata" in response_data:
        with st.expander("📊 Usage Metadata (TOKEN INFO!)", expanded=True):
            st.json(response_data["usage_metadata"])

    if "response_metadata" in response_data:
        with st.expander("📊 Response Metadata", expanded=True):
            st.json(response_data["response_metadata"])

    # Detailed view of each key
    st.subheader("🔍 Detailed Key Analysis")

    for key, value in response_data.items():
        with st.expander(
            f"📂 Key: '{key}' ({type(value).__name__})", expanded=False
        ):
            _display_value_analysis(key, value)


def _display_object_response(response_data):
    """Display analysis for object responses"""

    st.subheader("🔧 Object Attributes")

    # Special handling for LangChain message objects
    if hasattr(response_data, 'content'):
        st.subheader("💬 Message Content")
        st.text_area("Content", str(response_data.content), height=100, disabled=True)

        if hasattr(response_data, 'response_metadata'):
            with st.expander("📊 Response Metadata", expanded=True):
                st.json(_make_json_serializable(response_data.response_metadata))

        if hasattr(response_data, 'usage_metadata'):
            with st.expander("📊 Usage Metadata (TOKEN INFO!)", expanded=True):
                st.json(_make_json_serializable(response_data.usage_metadata))

    # Get all non-private attributes
    attributes = []
    for attr_name in dir(response_data):
        if not attr_name.startswith("_"):
            try:
                attr_value = getattr(response_data, attr_name)
                if not callable(attr_value):
                    attributes.append({
                        "Attribute": attr_name,
                        "Type": type(attr_value).__name__,
                        "Length": len(attr_value) if hasattr(attr_value, "__len__") else "N/A",
                        "Preview": (
                            str(attr_value)[:100] + "..."
                            if len(str(attr_value)) > 100
                            else str(attr_value)
                        ),
                    })
            except Exception:
                # Skip attributes that cause errors but note them
                attributes.append({
                    "Attribute": attr_name,
                    "Type": "Error",
                    "Length": "N/A",
                    "Preview": "<Error accessing attribute>",
                })

    if attributes:
        st.dataframe(attributes, use_container_width=True)

        # Detailed view of each attribute
        st.subheader("🔍 Detailed Attribute Analysis")

        for attr_info in attributes:
            attr_name = attr_info["Attribute"]
            if attr_info["Type"] != "Error":
                try:
                    attr_value = getattr(response_data, attr_name)

                    with st.expander(f"📂 Attribute: '{attr_name}' ({type(attr_value).__name__})", expanded=False):
                        _display_value_analysis(attr_name, attr_value)
                except Exception as e:
                    with st.expander(f"📂 Attribute: '{attr_name}' (Error)", expanded=False):
                        st.error(f"Error accessing attribute: {e}")
    else:
        st.warning("No accessible attributes found")


def _display_simple_response(response_data):
    """Display analysis for simple responses"""

    st.subheader("📄 Simple Response")
    st.code(str(response_data), language="python")


def _display_value_analysis(key, value):
    """Display detailed analysis of a specific value"""

    st.write(f"**Type:** `{type(value).__name__}`")

    if hasattr(value, "__len__"):
        st.write(f"**Length:** {len(value)}")

    # Special handling for different types
    if isinstance(value, str):
        st.write("**Content:**")
        st.text_area(f"String content for {key}", value, height=100, disabled=True)

    elif isinstance(value, list):
        st.write(f"**List with {len(value)} items:**")

        for i, item in enumerate(value):
            with st.expander(f"Item {i+1}: {type(item).__name__}", expanded=False):
                if hasattr(item, "__dict__"):
                    # Object with attributes
                    st.write("**Object attributes:**")
                    for attr_name in dir(item):
                        if not attr_name.startswith("_"):
                            try:
                                attr_value = getattr(item, attr_name)
                                if not callable(attr_value):
                                    st.write(
                                        f"- **{attr_name}:** {type(attr_value).__name__}"
                                    )
                                    if isinstance(attr_value, (str, int, float, bool)):
                                        st.code(str(attr_value))
                                    elif hasattr(attr_value, "__dict__"):
                                        st.write(
                                            f"  (Complex object: {type(attr_value).__name__})"
                                        )
                            except Exception:
                                continue
                else:
                    st.code(str(item))

    elif isinstance(value, dict):
        st.write("**Dictionary contents:**")
        for sub_key, sub_value in value.items():
            st.write(f"- **{sub_key}:** `{type(sub_value).__name__}`")
            if isinstance(sub_value, (str, int, float, bool)):
                st.code(str(sub_value))

    else:
        # Other types
        if hasattr(value, "__dict__"):
            st.write("**Object attributes:**")
            for attr_name in dir(value):
                if not attr_name.startswith("_"):
                    try:
                        attr_value = getattr(value, attr_name)
                        if not callable(attr_value):
                            st.write(f"- **{attr_name}:** {type(attr_value).__name__}")
                    except Exception:
                        continue

        st.write("**String representation:**")
        st.code(str(value), language="python")


def _make_json_serializable(obj):
    """Convert object to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Handle LangChain message objects specially
        if hasattr(obj, 'content') and hasattr(obj, 'response_metadata'):
            try:
                return {
                    "_type": type(obj).__name__,
                    "content": str(obj.content),
                    "response_metadata": _make_json_serializable(obj.response_metadata),
                    "usage_metadata": _make_json_serializable(getattr(obj, 'usage_metadata', None)),
                    "_additional_attributes": {
                        attr: _safe_get_attribute(obj, attr)
                        for attr in dir(obj)
                        if not attr.startswith("_") and attr not in ['content', 'response_metadata', 'usage_metadata']
                        and not callable(getattr(obj, attr, None))
                    }
                }
            except Exception:
                return f"<LangChain {type(obj).__name__} object: {str(obj)[:200]}>"

        # Convert complex objects to string representation
        elif hasattr(obj, "__dict__"):
            try:
                return {
                    "_type": type(obj).__name__,
                    "_attributes": {
                        attr: _safe_get_attribute(obj, attr)
                        for attr in dir(obj)
                        if not attr.startswith("_") and not callable(getattr(obj, attr, None))
                    },
                }
            except Exception:
                return f"<{type(obj).__name__} object: {str(obj)[:200]}>"
        else:
            return str(obj)


def _safe_get_attribute(obj, attr_name):
    """Safely get an attribute value without causing errors"""
    try:
        attr_value = getattr(obj, attr_name)
        # Don't recursively serialize complex objects to avoid infinite loops
        if isinstance(attr_value, (str, int, float, bool, type(None))):
            return attr_value
        elif isinstance(attr_value, (list, dict)):
            return _make_json_serializable(attr_value)
        else:
            return f"<{type(attr_value).__name__} object>"
    except Exception:
        return f"<Error accessing {attr_name}>"
