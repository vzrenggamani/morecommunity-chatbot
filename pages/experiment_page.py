import streamlit as st
from datetime import datetime
from utils.llm_utils import get_llm_instance, load_llm_and_retriever
from langchain.callbacks.base import BaseCallbackHandler


class GoogleUsageLogger(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        print("LLM Usage Metadata:", response.response_metadata)


def show_qa_vs_llm_experiment_page():
    """Experimental page to compare QA chain vs Direct LLM responses"""

    # Add logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🧪 QA Chain vs Direct LLM Experiment")

    st.markdown("---")

    st.markdown(
        """
    **Purpose:** This page compares responses from the QA chain vs direct LLM calls to understand
    why the QA chain doesn't provide full metadata like direct LLM calls.

    **Hypothesis:** The QA chain processes and filters the raw LLM response, potentially stripping metadata.
    """
    )

    # Initialize session state
    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = []

    # Test query section
    st.header("🧪 Experiment Setup")

    default_question = "Apakah wajar saya marah ketika kerabat tidak mendukung saya ketika di masa sulit dengan anak saya yang autis?"

    test_query = st.text_area(
        "Test Question:",
        value=default_question,
        height=80,
        help="Same question will be sent to both QA chain and direct LLM",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧪 Run Full Experiment", type="primary"):
            if test_query.strip():
                _run_full_experiment(test_query)

    with col2:
        if st.button("🗑️ Clear Results"):
            st.session_state.experiment_results = []
            st.success("✅ Results cleared!")

    # Display results
    if st.session_state.experiment_results:
        st.header("📊 Experiment Results")

        # Result selector
        result_options = [
            f"{i+1}. {result['timestamp']} - {result['query'][:40]}{'...' if len(result['query']) > 40 else ''}"
            for i, result in enumerate(st.session_state.experiment_results)
        ]

        selected_idx = st.selectbox(
            "Select experiment result:",
            range(len(result_options)),
            format_func=lambda x: result_options[x],
        )

        if selected_idx is not None:
            selected_result = st.session_state.experiment_results[selected_idx]
            _display_experiment_analysis(selected_result)

    else:
        st.info(
            "💡 Click 'Run Full Experiment' to compare QA chain vs direct LLM responses."
        )


def _run_full_experiment(test_query):
    """Run the complete experiment comparing QA chain vs direct LLM"""

    experiment_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": test_query,
        "qa_chain_result": None,
        "direct_llm_result": None,
        "experiment_log": [],
    }

    with st.spinner("🧪 Running experiment..."):

        # Step 1: Test QA Chain
        st.info("📋 Step 1: Testing QA Chain...")
        try:
            qa_chain = load_llm_and_retriever()
            qa_response = qa_chain.invoke(
                {"query": test_query}, config={"callbacks": [GoogleUsageLogger()]}
            )

            qa_analysis = _analyze_qa_response(qa_response)
            experiment_result["qa_chain_result"] = qa_analysis
            experiment_result["experiment_log"].append("✅ QA Chain call successful")

        except Exception as e:
            experiment_result["qa_chain_result"] = {"error": str(e)}
            experiment_result["experiment_log"].append(f"❌ QA Chain error: {e}")

        # Step 2: Test Direct LLM
        st.info("🤖 Step 2: Testing Direct LLM...")
        try:
            llm = get_llm_instance()
            direct_response = llm.invoke(test_query)

            direct_analysis = _analyze_direct_response(direct_response)
            experiment_result["direct_llm_result"] = direct_analysis
            experiment_result["experiment_log"].append("✅ Direct LLM call successful")

        except Exception as e:
            experiment_result["direct_llm_result"] = {"error": str(e)}
            experiment_result["experiment_log"].append(f"❌ Direct LLM error: {e}")

        except Exception as e:
            experiment_result["llm_generate_result"] = {"error": str(e)}
            experiment_result["experiment_log"].append(f"❌ LLM Generate error: {e}")

        # Step 4: Test QA Chain Internals
        st.info("🔍 Step 4: Investigating QA Chain Internals...")
        try:
            qa_internal_analysis = _investigate_qa_chain_internals(qa_chain, test_query)
            experiment_result["qa_internal_analysis"] = qa_internal_analysis
            experiment_result["experiment_log"].append(
                "✅ QA Chain internal analysis complete"
            )

        except Exception as e:
            experiment_result["qa_internal_analysis"] = {"error": str(e)}
            experiment_result["experiment_log"].append(
                f"❌ QA Chain internal analysis error: {e}"
            )

    # Store result
    st.session_state.experiment_results.insert(0, experiment_result)
    if len(st.session_state.experiment_results) > 5:
        st.session_state.experiment_results = st.session_state.experiment_results[:5]

    st.success("✅ Experiment completed! Check results below.")


def _analyze_qa_response(qa_response):
    """Analyze QA chain response structure"""
    analysis = {
        "type": str(type(qa_response)),
        "has_metadata": False,
        "has_usage_metadata": False,
        "has_source_documents": False,
        "structure": {},
    }

    try:
        if isinstance(qa_response, dict):
            analysis["structure"] = {
                "keys": list(qa_response.keys()),
                "key_types": {k: str(type(v)) for k, v in qa_response.items()},
            }

            # Check for specific keys
            if "result" in qa_response:
                analysis["has_result"] = True
                analysis["result_type"] = str(type(qa_response["result"]))
                analysis["result_preview"] = str(qa_response["result"])

            if "source_documents" in qa_response:
                analysis["has_source_documents"] = True
                analysis["source_documents_count"] = len(
                    qa_response["source_documents"]
                )

            # Look for any metadata
            for key, value in qa_response.items():
                if "metadata" in key.lower():
                    analysis["has_metadata"] = True
                    analysis[f"{key}_type"] = str(type(value))
                    analysis[f"{key}_content"] = str(value)

        analysis["full_str"] = str(qa_response)

    except Exception as e:
        analysis["analysis_error"] = str(e)

    return analysis


def _analyze_direct_response(direct_response):
    """Analyze direct LLM response structure"""
    analysis = {
        "type": str(type(direct_response)),
        "has_content": False,
        "has_response_metadata": False,
        "has_usage_metadata": False,
        "attributes": [],
    }

    try:
        # Check for specific attributes
        for attr in ["content", "response_metadata", "usage_metadata"]:
            if hasattr(direct_response, attr):
                attr_value = getattr(direct_response, attr)
                analysis[f"has_{attr}"] = True
                analysis[f"{attr}_type"] = str(type(attr_value))
                analysis[f"{attr}_content"] = str(attr_value)

        # Get all available attributes
        analysis["attributes"] = [
            attr for attr in dir(direct_response) if not attr.startswith("_")
        ]

        analysis["full_str"] = str(direct_response)

    except Exception as e:
        analysis["analysis_error"] = str(e)

    return analysis


def _analyze_generate_response(generate_response):
    """Analyze LLM generate response structure"""
    analysis = {
        "type": str(type(generate_response)),
        "attributes": [],
        "structure_analysis": {},
    }

    try:
        # Check structure
        analysis["attributes"] = [
            attr for attr in dir(generate_response) if not attr.startswith("_")
        ]

        # Try to access generations if it exists
        if hasattr(generate_response, "generations"):
            generations = generate_response.generations
            analysis["has_generations"] = True
            analysis["generations_type"] = str(type(generations))
            analysis["generations_length"] = (
                len(generations) if hasattr(generations, "__len__") else "N/A"
            )

            # Try to access first generation
            if hasattr(generations, "__getitem__") and len(generations) > 0:
                first_gen = generations[0]
                analysis["first_generation_type"] = str(type(first_gen))

                if hasattr(first_gen, "__getitem__") and len(first_gen) > 0:
                    first_message = first_gen[0]
                    analysis["first_message_type"] = str(type(first_message))
                    analysis["first_message_attributes"] = [
                        attr for attr in dir(first_message) if not attr.startswith("_")
                    ]

                    # Check for metadata in the message
                    for attr in ["response_metadata", "usage_metadata"]:
                        if hasattr(first_message, attr):
                            attr_value = getattr(first_message, attr)
                            analysis[f"message_{attr}"] = str(attr_value)

        # Check for llm_output
        if hasattr(generate_response, "llm_output"):
            llm_output = generate_response.llm_output
            analysis["has_llm_output"] = True
            analysis["llm_output_type"] = str(type(llm_output))
            analysis["llm_output_content"] = str(llm_output)

        analysis["full_str"] = str(generate_response)

    except Exception as e:
        analysis["analysis_error"] = str(e)

    return analysis


def _investigate_qa_chain_internals(qa_chain, test_query):
    """Investigate QA chain internal structure to understand metadata loss"""
    analysis = {
        "qa_chain_type": str(type(qa_chain)),
        "qa_chain_attributes": [],
        "llm_investigation": {},
        "retriever_investigation": {},
    }

    try:
        # Analyze QA chain structure
        analysis["qa_chain_attributes"] = [
            attr for attr in dir(qa_chain) if not attr.startswith("_")
        ]

        # Try to access the internal LLM
        if hasattr(qa_chain, "llm"):
            llm = qa_chain.llm
            analysis["llm_investigation"] = {
                "type": str(type(llm)),
                "attributes": [attr for attr in dir(llm) if not attr.startswith("_")],
            }

            # Try calling the internal LLM directly
            try:
                direct_llm_response = llm.invoke(test_query)
                analysis["llm_investigation"]["direct_call_result"] = {
                    "type": str(type(direct_llm_response)),
                    "has_usage_metadata": hasattr(
                        direct_llm_response, "usage_metadata"
                    ),
                    "usage_metadata": str(
                        getattr(direct_llm_response, "usage_metadata", "Not found")
                    ),
                }
            except Exception as e:
                analysis["llm_investigation"]["direct_call_error"] = str(e)

        # Try to access the retriever
        if hasattr(qa_chain, "retriever"):
            retriever = qa_chain.retriever
            analysis["retriever_investigation"] = {
                "type": str(type(retriever)),
                "attributes": [
                    attr for attr in dir(retriever) if not attr.startswith("_")
                ],
            }

        # Check chain type and configuration
        if hasattr(qa_chain, "chain_type"):
            analysis["chain_type"] = qa_chain.chain_type

        if hasattr(qa_chain, "return_source_documents"):
            analysis["return_source_documents"] = qa_chain.return_source_documents

    except Exception as e:
        analysis["investigation_error"] = str(e)

    return analysis


def _display_experiment_analysis(result):
    """Display comprehensive experiment analysis"""

    st.subheader("🧪 Experiment Analysis")
    st.caption(f"Executed at: {result['timestamp']}")

    # Show experiment log
    with st.expander("📋 Experiment Log", expanded=False):
        for log_entry in result["experiment_log"]:
            st.write(log_entry)

    # Create tabs for different analyses
    tabs = st.tabs(
        [
            "🔍 Comparison Overview",
            "📋 QA Chain",
            "🤖 Direct LLM",
            "📊 LLM Generate",
            "🔬 Internal Analysis",
        ]
    )

    # Tab 1: Comparison Overview
    with tabs[0]:
        st.subheader("📊 Metadata Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**QA Chain**")
            qa_result = result.get("qa_chain_result", {})
            st.write(f"- Has metadata: {qa_result.get('has_metadata', False)}")
            st.write(
                f"- Has usage metadata: {qa_result.get('has_usage_metadata', False)}"
            )

        with col2:
            st.write("**Direct LLM**")
            direct_result = result.get("direct_llm_result", {})
            st.write(
                f"- Has response metadata: {direct_result.get('has_response_metadata', False)}"
            )
            st.write(
                f"- Has usage metadata: {direct_result.get('has_usage_metadata', False)}"
            )

        with col3:
            st.write("**LLM Generate**")
            generate_result = result.get("llm_generate_result", {})
            st.write(
                f"- Has llm_output: {generate_result.get('has_llm_output', False)}"
            )
            st.write(
                f"- Has generations: {generate_result.get('has_generations', False)}"
            )

        # Key findings
        st.subheader("🎯 Key Findings")

        findings = []

        if not qa_result.get("has_usage_metadata", False) and direct_result.get(
            "has_usage_metadata", False
        ):
            findings.append(
                "✅ **Direct LLM has usage metadata, QA Chain doesn't** - QA Chain likely strips metadata during processing"
            )

        if qa_result.get("has_source_documents", False):
            findings.append(
                "✅ **QA Chain adds source documents** - This is RAG-specific functionality"
            )

        if qa_result.get("has_result", False):
            findings.append(
                "✅ **QA Chain wraps response in 'result' key** - This suggests response processing/transformation"
            )

        for finding in findings:
            st.write(finding)

        if not findings:
            st.write(
                "🤔 No clear patterns identified. Check individual tabs for details."
            )

    # Tab 2: QA Chain Analysis
    with tabs[1]:
        if "qa_chain_result" in result:
            st.json(result["qa_chain_result"])
        else:
            st.warning("No QA Chain result available")

    # Tab 3: Direct LLM Analysis
    with tabs[2]:
        if "direct_llm_result" in result:
            st.json(result["direct_llm_result"])
        else:
            st.warning("No Direct LLM result available")

    # Tab 4: LLM Generate Analysis
    with tabs[3]:
        if "llm_generate_result" in result:
            st.json(result["llm_generate_result"])
        else:
            st.warning("No LLM Generate result available")

    # Tab 5: Internal Analysis
    with tabs[4]:
        if "qa_internal_analysis" in result:
            st.subheader("🔬 QA Chain Internal Structure")
            st.json(result["qa_internal_analysis"])
        else:
            st.warning("No internal analysis available")
