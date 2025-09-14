# run.py

from src.pipeline import dp_training_deployment_pipeline

if __name__ == "__main__":
    print("Initiating deployment pipeline run...")
    
    # Run the pipeline with caching disabled for the first clean run.
    # On subsequent runs, you can remove the .with_options() part.
    dp_training_deployment_pipeline.with_options(enable_cache=False)()
    
    print("Pipeline run finished.")