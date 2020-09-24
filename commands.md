```
nohup python train_smart_batching.py --dataset_filepath "article_descriptions_train.pickle" --device_no 2 --truncation "head-and-tail" &> nohup_article_descriptions_full_smart_batching.out &

nohup python train_smart_batching.py --no_binary_classifier --dataset_filepath "article_headlines_train.pickle" --device_no 1 --truncation "head-and-tail" &> nohup_article_headlines_LCR_full_smart_batching.out &

nohup python train_smart_batching.py --binary_classifier --data_1 0 --data_2 1 --dataset_filepath "article_descriptions_train.pickle" --device_no 2 --truncation "head-and-tail" &> nohup_article_descriptions_LC_full_smart_batching.out &


nohup python train_smart_batching.py --no_binary_classifier --dataset_filepath "/data/madhu/article_headlines_train.pickle" --device_no 1 --truncation "head-and-tail" &> nohup_article_headlines_LCR_full_smart_batching.out &

```