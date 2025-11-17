"""
🚀 LoRA Fine-Tuning Interface
User-friendly Streamlit app for fine-tuning LLMs with LoRA on Mac (MPS)
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import streamlit as st
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, Dataset, DatasetDict

# Set page config
st.set_page_config(
    page_title="Fine-Tuning Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS


def load_css() -> None:
    """Load custom CSS from external file"""
    css_file = os.path.join(os.path.dirname(__file__), "style", "main.css")
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ CSS file not found!")


load_css()

# Initialize session state
if "training_started" not in st.session_state:
    st.session_state.training_started = False
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "device" not in st.session_state:
    st.session_state.device = None
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None


def check_mps_availability() -> Tuple[torch.device, bool, str]:
    """Check if MPS is available and return device info"""
    mps_is_available = torch.backends.mps.is_available()

    if mps_is_available:
        dev = torch.device("mps")
        msg = "✅ MPS (Apple Silicon) available"
    else:
        dev = torch.device("cpu")
        msg = "⚠️ MPS not available, using CPU (slower)"

    return dev, mps_is_available, msg


def find_target_modules(
    base_model: Union[PreTrainedModel, PeftModel],
    exclude_names: Optional[Set[str]] = None,
) -> List[str]:
    """Identify candidate module names to apply LoRA"""
    if exclude_names is None:
        exclude_names = {"lm_head", "embed_tokens", "wte", "wpe", "ln_f"}

    found_modules: Set[str] = set()
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = name.split(".")[-1]
            if module_name:
                found_modules.add(module_name)

    return list(found_modules - exclude_names)


def load_model_and_tokenizer(
    model_id: str, target_device: torch.device
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer"""
    with st.spinner(f"Loading model {model_id}..."):
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        if loaded_tokenizer.pad_token is None:
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": target_device},
            trust_remote_code=True,
        )

    return loaded_model, loaded_tokenizer


def prepare_dataset(
    ds_name: str, n_samples: int, tok: PreTrainedTokenizerBase, seq_max_length: int
) -> Tuple[Dataset, Dataset, int]:
    """Load and tokenize dataset"""
    with st.spinner(f"Loading dataset {ds_name}..."):
        dataset = load_dataset(ds_name, split=f"train[:{n_samples}]")

        def tokenize_function(examples: Dict[str, Any]) -> Union[Dict[str, Any], Any]:
            return tok(
                examples["text"],
                truncation=True,
                max_length=seq_max_length,
                padding="max_length",
                return_tensors=None,
            )

        # Get column names - handle different dataset types
        columns_to_remove: Optional[List[str]] = None
        if hasattr(dataset, "column_names"):
            cols = dataset.column_names  # type: ignore
            # Convert to list if it's a list/tuple
            if isinstance(cols, (list, tuple)):
                columns_to_remove = list(cols)

        tokenized_dataset = dataset.map(  # type: ignore
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
        )

        # Handle different dataset types
        if isinstance(tokenized_dataset, Dataset):
            split_ds = tokenized_dataset.train_test_split(
                test_size=0.1, seed=42)
        elif isinstance(tokenized_dataset, DatasetDict):
            # Already split
            split_ds = tokenized_dataset
        else:
            # Fallback: try to call train_test_split
            split_ds = tokenized_dataset.train_test_split(
                test_size=0.1, seed=42
            )  # type: ignore

    return split_ds["train"], split_ds["test"], n_samples  # type: ignore


def create_lora_config(
    r: int, alpha: int, dropout: float, modules_to_target: List[str]
) -> LoraConfig:
    """Create LoRA configuration"""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=modules_to_target,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def plot_training_metrics(training_history: List[Dict[str, Any]]) -> go.Figure:
    """Create interactive training metrics plot"""
    losses_train = [entry["loss"]
                    for entry in training_history if "loss" in entry]
    losses_eval = [
        entry["eval_loss"] for entry in training_history if "eval_loss" in entry
    ]

    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("Training Loss", "Validation Loss")
    )

    if losses_train:
        figure.add_trace(
            go.Scatter(
                y=losses_train,
                mode="lines",
                name="Training Loss",
                line=dict(color="#667eea", width=2),
            ),
            row=1,
            col=1,
        )

    if losses_eval:
        figure.add_trace(
            go.Scatter(
                y=losses_eval,
                mode="lines",
                name="Validation Loss",
                line=dict(color="#f093fb", width=2),
            ),
            row=1,
            col=2,
        )

    figure.update_layout(height=400, showlegend=True,
                         title_text="Training Metrics")

    return figure


# =============================================================================
# SIDEBAR - Configuration
# =============================================================================

st.sidebar.markdown("# ⚙️ Configuration")

# Device check
device, mps_available, status_msg = check_mps_availability()
st.session_state.device = device

st.sidebar.info(status_msg)
st.sidebar.markdown(f"**PyTorch:** {torch.__version__}")

st.sidebar.markdown("---")

# Model selection
st.sidebar.markdown("### 🤖 Model")
model_selection_type = st.sidebar.radio(
    "Selection mode",
    ["Preset", "Custom"],
    horizontal=True,
    help="Choose a preset model or enter a custom path"
)

if model_selection_type == "Preset":
    model_name = st.sidebar.selectbox(
        "Select base model",
        [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        help="Smaller models (1B) are faster but less capable",
    )
else:
    model_name = st.sidebar.text_input(
        "HuggingFace model path",
        value="",
        help="Enter the model path (e.g., 'meta-llama/Llama-3.2-1B' or 'gpt2')",
        placeholder="username/model-name"
    )

# Dataset selection
st.sidebar.markdown("### 📊 Dataset")
dataset_selection_type = st.sidebar.radio(
    "Dataset selection mode",
    ["Preset", "Custom"],
    horizontal=True,
    help="Choose a preset dataset or enter a custom path",
    key="dataset_radio"
)

if dataset_selection_type == "Preset":
    dataset_name = st.sidebar.selectbox(
        "Select dataset",
        ["imdb", "wikitext-2-raw-v1", "tiny_shakespeare"],
        help="IMDB: movie reviews, WikiText: Wikipedia text",
    )
else:
    dataset_name = st.sidebar.text_input(
        "HuggingFace dataset path",
        value="",
        help="Enter the dataset path (e.g., 'squad', 'glue/mrpc', 'wikipedia')",
        placeholder="username/dataset-name or dataset-name"
    )

num_samples = st.sidebar.slider(
    "Number of samples",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="More samples = longer training but better results",
)

max_length = st.sidebar.slider(
    "Max sequence length",
    min_value=128,
    max_value=1024,
    value=512,
    step=128,
    help="Longer sequences require more memory",
)

# LoRA configuration
st.sidebar.markdown("### 🔧 LoRA Parameters")
lora_r = st.sidebar.slider(
    "Rank (r)",
    min_value=4,
    max_value=64,
    value=16,
    step=4,
    help="LoRA matrix dimension. Higher = more parameters",
)

lora_alpha = st.sidebar.slider(
    "Alpha",
    min_value=8,
    max_value=128,
    value=32,
    step=8,
    help="Scaling factor. Generally alpha = 2 × r",
)

lora_dropout = st.sidebar.slider(
    "Dropout",
    min_value=0.0,
    max_value=0.2,
    value=0.05,
    step=0.05,
    help="Dropout for regularization",
)

# Training configuration
st.sidebar.markdown("### 🎯 Training")
num_epochs = st.sidebar.slider(
    "Epochs",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of complete passes through the dataset",
)

batch_size = st.sidebar.slider(
    "Batch size",
    min_value=1,
    max_value=16,
    value=4,
    step=1,
    help="Reduce if you have memory issues",
)

gradient_accumulation = st.sidebar.slider(
    "Gradient accumulation steps",
    min_value=1,
    max_value=16,
    value=4,
    step=1,
    help="Simulates larger batch sizes",
)

learning_rate = st.sidebar.select_slider(
    "Learning rate",
    options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
    value=2e-4,
    help="Learning speed",
)

# Output name
output_name = st.sidebar.text_input(
    "Output model name",
    value="my-lora-model",
    help="Name to save the fine-tuned model",
)

# =============================================================================
# MAIN AREA
# =============================================================================


st.title("🚀 LoRA Fine-Tuning Studio")
st.subheader("Fine-tuning LLMs on Mac with LoRA and MPS")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Setup", "🎓 Training", "🧪 Testing", "💾 Export"])

# =============================================================================
# TAB 1: SETUP
# =============================================================================

with tab1:
    st.markdown("## 📋 Setup and Preparation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Model")
        st.info(f"**Selected model:** `{model_name}`")

        if st.button(
            "🔄 Load Model and Tokenizer", type="primary", use_container_width=True
        ):
            try:
                model, tokenizer = load_model_and_tokenizer(model_name, device)
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True

                # Model info
                total_params = sum(p.numel() for p in model.parameters())

                st.success("✅ Model loaded successfully!")
                st.metric("Total parameters", f"{total_params:,}")

            except (ValueError, OSError, RuntimeError) as e:
                st.error(f"❌ Loading error: {str(e)}")
            except ImportError as e:
                st.error(
                    f"❌ Import error: {str(e)} - Check dependencies")

    with col2:
        st.markdown("### 📊 Dataset")
        st.info(
            f"**Dataset:** `{dataset_name}`\n\n**Samples:** {num_samples}")

        if st.button("📥 Load Dataset", type="primary", use_container_width=True):
            if not st.session_state.model_loaded:
                st.warning("⚠️ Load the model first!")
            else:
                try:
                    train_ds, eval_ds, total = prepare_dataset(
                        dataset_name,
                        num_samples,
                        st.session_state.tokenizer,
                        max_length,
                    )
                    st.session_state.train_dataset = train_ds
                    st.session_state.eval_dataset = eval_ds
                    st.session_state.dataset_loaded = True

                    st.success("✅ Dataset loaded and tokenized!")
                    col_a, col_b = st.columns(2)

                    col_a.metric("Training samples", len(train_ds))
                    col_b.metric("Validation samples", len(eval_ds))

                except (ValueError, KeyError, OSError) as e:
                    st.error(f"❌ Error: {str(e)}")
                except ImportError as e:
                    st.error(f"❌ Import error: {str(e)}")

    st.markdown("---")

    # LoRA Configuration
    if st.session_state.model_loaded:
        st.markdown("### ⚙️ Apply LoRA")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rank", lora_r)
        col2.metric("Alpha", lora_alpha)
        col3.metric("Dropout", f"{lora_dropout:.2f}")
        col4.metric("Batch Size", batch_size)

        if st.button(
            "🔧 Apply LoRA to Model", type="primary", use_container_width=True
        ):
            try:
                with st.spinner("Applying LoRA..."):
                    # Find target modules
                    target_modules = find_target_modules(
                        st.session_state.model)
                    st.info(f"**Target modules:** {', '.join(target_modules)}")

                    # Create LoRA config
                    lora_config = create_lora_config(
                        lora_r, lora_alpha, lora_dropout, target_modules
                    )

                    # Apply LoRA
                    st.session_state.model = get_peft_model(
                        st.session_state.model, lora_config
                    )

                    # Enable optimizations
                    # type: ignore[attr-defined]
                    st.session_state.model.gradient_checkpointing_enable()
                    # type: ignore[attr-defined]
                    st.session_state.model.enable_input_require_grads()
                    if hasattr(st.session_state.model, "config"):
                        # type: ignore[attr-defined]
                        st.session_state.model.config.use_cache = False

                    st.session_state.lora_applied = True

                st.success("✅ LoRA applied successfully!")

                # Show trainable parameters
                st.markdown("#### 📊 Parameters")
                trainable = sum(
                    p.numel()
                    for p in st.session_state.model.parameters()
                    if p.requires_grad
                )
                total = sum(p.numel()
                            for p in st.session_state.model.parameters())

                col1, col2, col3 = st.columns(3)
                col1.metric("Total", f"{total:,}")
                col2.metric("Trainable", f"{trainable:,}")
                col3.metric("% Trainable", f"{(trainable/total*100):.2f}%")

            except (ValueError, RuntimeError, AttributeError) as e:
                st.error(f"❌ Error: {str(e)}")
            except ImportError as e:
                st.error(
                    f"❌ PEFT error: {str(e)} - Check PEFT installation")

# =============================================================================
# TAB 2: TRAINING
# =============================================================================

with tab2:
    st.markdown("## 🎓 Model Training")

    if not st.session_state.get("lora_applied", False):
        st.warning("⚠️ Complete the setup in the Setup tab first!")
    else:
        # Training configuration summary
        st.markdown("### 📋 Training Configuration")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epochs", num_epochs)
        col2.metric("Batch Size", batch_size)
        col3.metric("Gradient Acc.", gradient_accumulation)
        col4.metric("Learning Rate", f"{learning_rate:.0e}")

        effective_batch = batch_size * gradient_accumulation
        st.info(f"**Effective Batch Size:** {effective_batch}")

        st.markdown("---")

        # Start training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state.training_started = True

            try:
                # Prepare data collator
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=st.session_state.tokenizer, mlm=False
                )

                # Training arguments
                training_args = TrainingArguments(
                    output_dir=f"./{output_name}-finetuned",
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation,
                    learning_rate=learning_rate,
                    warmup_steps=100,
                    gradient_checkpointing=True,
                    fp16=False,
                    bf16=False,
                    logging_steps=10,
                    report_to="none",
                    eval_strategy="steps",
                    eval_steps=50,
                    save_strategy="steps",
                    save_steps=100,
                    save_total_limit=3,
                    load_best_model_at_end=True,
                    remove_unused_columns=False,
                    seed=42,
                )

                # Create trainer
                trainer = Trainer(
                    model=st.session_state.model,
                    args=training_args,
                    train_dataset=st.session_state.train_dataset,
                    eval_dataset=st.session_state.eval_dataset,
                    data_collator=data_collator,
                )

                st.session_state.trainer = trainer

                # Training
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🔄 Training in progress...")

                train_result = trainer.train()

                progress_bar.progress(100)
                status_text.text("✅ Training completed!")

                st.session_state.training_complete = True
                st.session_state.train_result = train_result

                # Show results
                st.success("🎉 Training completed successfully!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Training Loss",
                            f"{train_result.training_loss:.4f}")
                col2.metric("Steps", train_result.global_step)
                col3.metric(
                    "Time (s)", f"{train_result.metrics['train_runtime']:.1f}")

                # Evaluate
                eval_results = trainer.evaluate()
                st.metric("Validation Loss",
                          f"{eval_results['eval_loss']:.4f}")

                # Plot metrics
                if len(trainer.state.log_history) > 0:
                    st.markdown("### 📊 Training Metrics")
                    fig = plot_training_metrics(trainer.state.log_history)
                    st.plotly_chart(fig, use_container_width=True)

            except (ValueError, RuntimeError, OSError) as e:
                st.error(f"❌ Training error: {str(e)}")
            except KeyboardInterrupt:
                st.warning("⚠️ Training interrupted by user")

        # Show metrics if training is complete
        if st.session_state.training_complete and st.session_state.trainer:
            st.markdown("---")
            st.markdown("### 📈 Post-Training Analysis")

            history = st.session_state.trainer.state.log_history
            train_losses = [e["loss"] for e in history if "loss" in e]
            eval_losses = [e["eval_loss"] for e in history if "eval_loss" in e]

            if train_losses and eval_losses:
                gap = eval_losses[-1] - train_losses[-1]

                col1, col2, col3 = st.columns(3)
                col1.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
                col2.metric("Final Eval Loss", f"{eval_losses[-1]:.4f}")
                col3.metric("Gap", f"{gap:.4f}")

                if gap < 0.1:
                    st.success("✅ Great! No significant overfitting")
                elif gap < 0.3:
                    st.warning("⚠️ Slight overfitting, but acceptable")
                else:
                    st.error(
                        "❌ Overfitting! Consider reducing epochs or increasing dropout"
                    )

# =============================================================================
# TAB 3: TESTING
# =============================================================================

with tab3:
    st.markdown("## 🧪 Fine-Tuned Model Testing")

    if not st.session_state.training_complete:
        st.warning("⚠️ Complete training first!")
    else:
        st.success("✅ Model ready for testing")

        # Generation parameters
        st.markdown("### ⚙️ Generation Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            max_new_tokens = st.slider("Max New Tokens", 20, 200, 100, 10)
        with col2:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        with col3:
            top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)

        st.markdown("---")

        # Text generation
        st.markdown("### 💬 Generate Text")

        prompt = st.text_area(
            "Enter prompt",
            value="This movie was absolutely",
            height=100,
            help="The model will complete this text",
        )

        col1, col2 = st.columns([1, 4])

        with col1:
            generate_btn = st.button(
                "🎯 Generate", type="primary", use_container_width=True
            )

        if generate_btn and prompt:
            try:
                with st.spinner("Generating..."):
                    # Set model to eval mode
                    st.session_state.model.eval()  # type: ignore[attr-defined]

                    # Tokenize
                    inputs = st.session_state.tokenizer(prompt, return_tensors="pt").to(  # type: ignore[operator]
                        device
                    )

                    # Generate
                    with torch.no_grad():
                        outputs = st.session_state.model.generate(  # type: ignore[attr-defined]
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True,
                            top_p=top_p,
                            top_k=50,
                            # type: ignore[attr-defined]
                            pad_token_id=st.session_state.tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                        )

                    generated_text = st.session_state.tokenizer.decode(  # type: ignore[attr-defined]
                        outputs[0], skip_special_tokens=True
                    )

                st.markdown("### 📝 Result")
                st.markdown(
                    f'<div class="success-box">{generated_text}</div>',
                    unsafe_allow_html=True,
                )

                # Show only generated part
                generated_only = generated_text[len(prompt):].strip()
                st.markdown("**Generated part:**")
                st.code(generated_only, language="text")

            except (ValueError, RuntimeError) as e:
                st.error(f"❌ Generation error: {str(e)}")

        st.markdown("---")

        # Quick test prompts
        st.markdown("### 🎲 Quick Tests")

        test_prompts = [
            "This movie was absolutely",
            "I really enjoyed",
            "The plot was",
            "The acting performance",
        ]

        if st.button("🔄 Generate quick examples"):
            for i, test_prompt in enumerate(test_prompts, 1):
                with st.expander(f"Example {i}: '{test_prompt}'"):
                    try:
                        inputs = st.session_state.tokenizer(  # type: ignore[operator]
                            test_prompt, return_tensors="pt"
                        ).to(device)

                        with torch.no_grad():
                            outputs = st.session_state.model.generate(  # type: ignore[attr-defined]
                                **inputs,
                                max_new_tokens=50,
                                temperature=0.7,
                                do_sample=True,
                                top_p=0.9,
                                # type: ignore[attr-defined]
                                pad_token_id=st.session_state.tokenizer.eos_token_id,
                            )

                        result = st.session_state.tokenizer.decode(  # type: ignore[attr-defined]
                            outputs[0], skip_special_tokens=True
                        )

                        st.write(result)

                    except (ValueError, RuntimeError) as e:
                        st.error(f"Error: {str(e)}")

# =============================================================================
# TAB 4: EXPORT
# =============================================================================

with tab4:
    st.markdown("## 💾 Export and Save")

    if not st.session_state.training_complete:
        st.warning("⚠️ Complete training first!")
    else:
        col1, col2 = st.columns(2)

        # Save LoRA adapter
        with col1:
            st.markdown("### 💾 Save LoRA Adapter")
            st.info("Save only LoRA adapters (~few MB)")

            adapter_path = f"./{output_name}-finetuned"
            st.text_input("Adapter path", adapter_path, disabled=True)

            if st.button("💾 Save Adapter", type="primary", use_container_width=True):
                try:
                    st.session_state.model.save_pretrained(
                        adapter_path
                    )  # type: ignore[attr-defined]
                    st.session_state.tokenizer.save_pretrained(
                        adapter_path
                    )  # type: ignore[attr-defined]

                    # Calculate size
                    total_size = sum(
                        os.path.getsize(os.path.join(adapter_path, f))
                        for f in os.listdir(adapter_path)
                    ) / (1024 * 1024)

                    st.success(
                        f"✅ Adapter saved! Size: {total_size:.1f} MB")

                    # Show loading code
                    with st.expander("📖 Code to load the adapter"):
                        st.code(
                            f"""
                            from transformers import AutoModelForCausalLM, AutoTokenizer
                            from peft import PeftModel

                            # Load base model
                            base_model = AutoModelForCausalLM.from_pretrained("{model_name}")

                            # Load LoRA adapter
                            model = PeftModel.from_pretrained(base_model, "{adapter_path}")

                            # Load tokenizer
                            tokenizer = AutoTokenizer.from_pretrained("{adapter_path}")
                            """,
                            language="python",
                        )

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Merge and save full model
        with col2:
            st.markdown("### 🔗 Merge and Save Full Model")
            st.info("Combine adapter with base model (full size)")

            merged_path = f"./{output_name}-merged"
            st.text_input("Merged path", merged_path, disabled=True)

            if st.button("🔗 Merge and Save", type="primary", use_container_width=True):
                try:
                    with st.spinner("Merging..."):
                        # type: ignore[attr-defined]
                        merged_model = st.session_state.model.merge_and_unload()
                        # type: ignore[attr-defined]
                        merged_model.save_pretrained(merged_path)
                        st.session_state.tokenizer.save_pretrained(
                            merged_path
                        )  # type: ignore[attr-defined]

                    # Calculate size
                    total_size = sum(
                        os.path.getsize(os.path.join(merged_path, f))
                        for f in os.listdir(merged_path)
                    ) / (1024 * 1024)

                    st.success(
                        f"✅ Merged model saved! Size: {total_size:.1f} MB"
                    )

                    # Show loading code
                    with st.expander("📖 Code to load the merged model"):
                        st.code(
                            f"""
                            from transformers import AutoModelForCausalLM, AutoTokenizer

                            # Load merged model (no PEFT needed)
                            model = AutoModelForCausalLM.from_pretrained("{merged_path}")
                            tokenizer = AutoTokenizer.from_pretrained("{merged_path}")
                            """,
                            language="python",
                        )

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        st.markdown("---")

        # Comparison
        st.markdown("### ⚖️ Adapter vs Merged Comparison")

        comparison_data = {
            "Feature": [
                "File size",
                "Inference speed",
                "Flexibility",
                "Recommended use",
            ],
            "LoRA Adapter": [
                "~few MB",
                "Slightly slower",
                "High (multiple adapters per base model)",
                "Development and experiments",
            ],
            "Merged Model": [
                "~full base model size",
                "Slightly faster",
                "Low (single model)",
                "Production deployment",
            ],
        }

        st.table(comparison_data)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 LoRA Fine-Tuning Studio | Made with ❤️ by <a href="https://justanotherai.company">Just Another AI Company</a></p>
        <p>Powered by 🤗 Transformers, PEFT, and Apple MPS</p>
    </div>
    """,
    unsafe_allow_html=True,
)
