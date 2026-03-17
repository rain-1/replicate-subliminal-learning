mkdir -p outputs/
for animal in dragons
do
    # Define the output file path as a variable
    OUTPUT_FILE="outputs/numbers-$animal.jsonl"

    # Check if the file already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $animal: $OUTPUT_FILE already exists."
        continue
    fi
    
    python data/generate-animal-numbers-data.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --animal $animal \
        --output $OUTPUT_FILE
done
