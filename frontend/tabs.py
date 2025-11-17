"""
Tab UI components for the main interface
"""

import os
import streamlit as st
import torch
from backend import (
    load_model_and_tokenizer,
    prepare_dataset,
    find_target_modules,
    create_lora_config,
    apply_lora_to_model,
    get_training_args,
    create_trainer,
    plot_training_metrics,
)


def render_setup_tab(
    model_name: str,
    dataset_name: str,
    num_samples: int,
    max_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    device: torch.device,
) -> None:
    """Render the Setup tab"""
    st.header("📋 Setup and Preparation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Model")
        st.info(f"**Selected model:** `{model_name}`", icon="ℹ️")

        if st.button(
            "Load Model and Tokenizer", type="primary", use_container_width=True
        ):
            try:
                with st.spinner(f"Loading model {model_name}..."):
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
                st.error(f"❌ Import error: {str(e)} - Check dependencies")

    with col2:
        st.subheader("📊 Dataset")
        st.info(
            f"**Dataset:** `{dataset_name}`\n\n**Samples:** {num_samples}", icon="ℹ️"
        )

        if st.button("Load Dataset", type="primary", use_container_width=True):
            if not st.session_state.model_loaded:
                st.warning("⚠️ Load the model first!")
            else:
                try:
                    with st.spinner(f"Loading dataset {dataset_name}..."):
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

    st.divider()

    # LoRA Configuration
    if st.session_state.model_loaded:
        st.subheader("⚙️ Apply LoRA")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rank", lora_r)
        col2.metric("Alpha", lora_alpha)
        col3.metric("Dropout", f"{lora_dropout:.2f}")
        col4.metric("Batch Size", batch_size)

        if st.button(
            "Apply LoRA to Model", type="primary", use_container_width=True, icon="🔧"
        ):
            try:
                with st.spinner("Applying LoRA..."):
                    # Find target modules
                    target_modules = find_target_modules(st.session_state.model)
                    st.info(f"**Target modules:** {', '.join(target_modules)}")

                    # Create LoRA config
                    lora_config = create_lora_config(
                        lora_r, lora_alpha, lora_dropout, target_modules
                    )

                    # Apply LoRA
                    st.session_state.model = apply_lora_to_model(
                        st.session_state.model, lora_config
                    )

                    st.session_state.lora_applied = True

                st.success("✅ LoRA applied successfully!")

                # Show trainable parameters
                st.subheader("📈 Parameters")
                trainable = sum(
                    p.numel()
                    for p in st.session_state.model.parameters()
                    if p.requires_grad
                )
                total = sum(p.numel() for p in st.session_state.model.parameters())

                col1, col2, col3 = st.columns(3)
                col1.metric("Total", f"{total:,}")
                col2.metric("Trainable", f"{trainable:,}")
                col3.metric("% Trainable", f"{(trainable/total*100):.2f}%")

            except (ValueError, RuntimeError, AttributeError) as e:
                st.error(f"❌ Error: {str(e)}")
            except ImportError as e:
                st.error(f"❌ PEFT error: {str(e)} - Check PEFT installation")


def render_training_tab(
    num_epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    output_name: str,
) -> None:
    """Render the Training tab"""
    st.header("🎓 Model Training")

    if not st.session_state.get("lora_applied", False):
        st.warning("⚠️ Complete the setup in the Setup tab first!")
    else:
        # Training configuration summary
        st.subheader("📋 Training Configuration")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epochs", num_epochs)
        col2.metric("Batch Size", batch_size)
        col3.metric("Gradient Acc.", gradient_accumulation)
        col4.metric("Learning Rate", f"{learning_rate:.0e}")

        effective_batch = batch_size * gradient_accumulation
        st.info(f"**Effective Batch Size:** {effective_batch}")

        st.divider()

        # Start training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            # Validate output_name
            if not output_name or output_name.strip() == "":
                st.error(
                    "❌ Please enter an output model name in the sidebar before starting training!"
                )
                return

            st.session_state.training_started = True

            try:
                # Training arguments
                training_args = get_training_args(
                    output_dir=f"./{output_name}-finetuned",
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation,
                    learning_rate=learning_rate,
                )

                # Create trainer
                trainer = create_trainer(
                    model=st.session_state.model,
                    tokenizer=st.session_state.tokenizer,
                    train_dataset=st.session_state.train_dataset,
                    eval_dataset=st.session_state.eval_dataset,
                    training_args=training_args,
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
                col1.metric("Training Loss", f"{train_result.training_loss:.4f}")
                col2.metric("Steps", train_result.global_step)
                col3.metric("Time (s)", f"{train_result.metrics['train_runtime']:.1f}")

                # Evaluate
                eval_results = trainer.evaluate()
                st.metric("Validation Loss", f"{eval_results['eval_loss']:.4f}")

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
            st.divider()
            st.subheader("📈 Post-Training Analysis")

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


def render_testing_tab(device: torch.device) -> None:
    """Render the Testing tab"""
    st.header("🧪 Fine-Tuned Model Testing")

    if not st.session_state.training_complete:
        st.warning("⚠️ Complete training first!")
    else:
        st.success("✅ Model ready for testing")

        # Generation parameters
        st.subheader("⚙️ Generation Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            max_new_tokens = st.slider("Max New Tokens", 20, 200, 100, 10)
        with col2:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        with col3:
            top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)

        st.divider()

        # Text generation
        st.subheader("💬 Generate Text")

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

                st.subheader("📝 Result")
                st.success(generated_text)

                # Show only generated part
                generated_only = generated_text[len(prompt) :].strip()
                st.markdown("**Generated part:**")
                st.code(generated_only, language="text")

            except (ValueError, RuntimeError) as e:
                st.error(f"❌ Generation error: {str(e)}")

        st.divider()

        # Quick test prompts
        st.subheader("🎲 Quick Tests")

        test_prompts = [
            "This movie was absolutely",
            "I really enjoyed",
            "The plot was",
            "The acting performance",
        ]

        if st.button(
            "Generate quick examples",
            type="primary",
            use_container_width=True,
            icon="✨",
        ):
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


def render_export_tab(model_name: str, output_name: str) -> None:
    """Render the Export tab"""
    st.header("💾 Export and Save")

    if not st.session_state.training_complete:
        st.warning("⚠️ Complete training first!")
    elif not output_name or output_name.strip() == "":
        st.error(
            "❌ Please enter an output model name in the sidebar before exporting!"
        )
    else:
        col1, col2 = st.columns(2)

        # Save LoRA adapter
        with col1:
            st.subheader("💾 Save LoRA Adapter")
            st.info("Save only LoRA adapters (~few MB)", icon="ℹ️")

            adapter_path = f"./{output_name}-finetuned"
            st.text_input(
                "Adapter path", adapter_path, disabled=True, key="adapter_path_input"
            )

            if st.button(
                "Save Adapter", type="primary", use_container_width=True, icon="💾"
            ):
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

                    st.success(f"✅ Adapter saved! Size: {total_size:.1f} MB")

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
            st.text_input(
                "Merged path", merged_path, disabled=True, key="merged_path_input"
            )

            if st.button(
                "Merge and Save", type="primary", use_container_width=True, icon="🔗"
            ):
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

                    st.success(f"✅ Merged model saved! Size: {total_size:.1f} MB")

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

        st.divider()

        # Comparison
        st.subheader("⚖️ Adapter vs Merged Comparison")

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
