from google.cloud import aiplatform

aiplatform.init(
    project="deep-learning-489620",
    location="us-west1",
    staging_bucket="gs://deep_learning_dataset/staging"
)

job = aiplatform.CustomJob.from_local_script(
    display_name="qwen3-streaming-tuning-v2",
    script_path="task.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest",
    requirements=[
        "transformers==4.40.2",
        "accelerate==0.29.3",
        "peft==0.10.0",
        "datasets==2.19.1",
        "bitsandbytes==0.43.1",
        "sentencepiece>=0.1.99",
        "tiktoken>=0.7.0",
        # "transformers==4.41.2",
        # "peft==0.11.1",
        # "accelerate==0.30.1",
        # "datasets==2.20.0",
        # "bitsandbytes==0.43.2",
        # "sentencepiece",
        # "tiktoken"
    ],
    replica_count=1,
    machine_type="n1-standard-8",               # T4 requires N1 machines
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=[
        "--train_manifest", "gs://deep_learning_dataset/train.jsonl",
        "--library_path", "gs://deep_learning_dataset/per_drug_data.json",
        "--output_dir", "gs://deep_learning_dataset/model_output"
    ]
)

job.run()