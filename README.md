# SetLLM
---

## Running the Pipeline from Scratch

### 1. Preprocess the Dataset
Run the following scripts to prepare the dataset:
```bash
cd data/
python split.py --dataset {YourDataset}
python convert.py --dataset {YourDataset}
```

### 2. Initialize the User Representation
We use the original implementation of [**LightGCN**](https://github.com/gusye1234/LightGCN-PyTorch) to obtain the initialization representations for **SetLLM**.  
During training, we swap the positions of users and items when constructing the training and test sets — that is, each line starts with an **item**, followed by all the **users** that have interacted with it — to implement an **item-oriented BPR loss**.  

After training, the resulting user and item embeddings are saved as **`init_user_weight.pt`** and **`init_item_weight.pt`**, respectively, and placed in:
```bash
model_weight/{YourDataset}/{YourBackbone}/stage_1/
```

### 3. Fine-tune SetLLM
Run the following command to fine-tune **SetLLM** on your dataset:
```bash
python Stage_2.py --dataset {YourDataset} --LLM_type {YourBackbone}
```

### 4. Apply DCPO Optimization
Run the **DCPO** process to further optimize the fine-tuned model:
```bash
python Stage_3_fin.py --dataset {YourDataset} --LLM_type {YourBackbone}
```

### 5. Generate Sets for Cold Items
Use the trained model to generate item sets for cold-start scenarios:
```bash
python predict.py --dataset {YourDataset} --LLM_type {YourBackbone}
```

### 6. Evaluate the Model
Evaluate the final model performance:
```bash
cd Evaluation/
python main.py --dataset {YourDataset} --LLM_type {YourBackbone}
```
