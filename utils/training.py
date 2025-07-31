import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from tqdm import tqdm


def bprmf_training(model, dataset, accelerator, collator, batch_size, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for batch in dataloader:
            items, pos_users, neg_users = batch
            optimizer.zero_grad()
            # TODO 这里可能之后还是使用user侧的bpr
            users = items.to(accelerator.device)
            pos_items = pos_users.to(accelerator.device)
            neg_items = neg_users.to(accelerator.device)
            loss = model.item_bpr_loss(users, pos_items, neg_items)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
        # if epoch % 10 == 0:
        #     accelerator.print(f"[BPRMF] Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return model

def dpo_training(model, train_dataset, test_dataset, collator, training_args):
    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
    )
    trainer.train()
    return model