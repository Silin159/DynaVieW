export OPENAI_API_KEY=""
export GEMINI_API_KEY=""

python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gpt-4o --eval-type char_num --seed 42
python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gpt-4o --eval-type char_attr --seed 42
python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gpt-4o --eval-type entity --seed 42
python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gpt-4o --eval-type location --seed 42
python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gpt-4o --eval-type time --seed 42

python vqa_consistency.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gpt-4o --eval-type char --seed 42
python vqa_consistency.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gpt-4o --eval-type location --seed 42
python vqa_consistency.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gpt-4o --eval-type style --seed 42


# python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gemini-2.5-pro --eval-type char_num --seed 42
# python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gemini-2.5-pro --eval-type char_attr --seed 42
# python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gemini-2.5-pro --eval-type entity --seed 42
# python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gemini-2.5-pro --eval-type location --seed 42
# python vqa_alignment.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gemini-2.5-pro --eval-type time --seed 42

# python vqa_consistency.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gemini-2.5-pro --eval-type char --seed 42
# python vqa_consistency.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gemini-2.5-pro --eval-type location --seed 42
# python vqa_consistency.py --dataset-name vwp --annotation-path ./annotations/vwp_test.json --model-name model_name --judge-model gemini-2.5-pro --eval-type style --seed 42