#imports
import json
import pandas as pd
import torch
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from transformers import DefaultDataCollator
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

#data preparation
data_file = open('../Data/citation_data.json')
lines = data_file.readlines()
json_lines = [json.loads(line) for line in lines]
cc_df = pd.DataFrame(json_lines)


cc_df['context'] = cc_df['text_before_explicit_citation'].str.cat([cc_df['explicit_citation'], cc_df['text_after_explicit_citation']], sep=' ')

dataframe = cc_df[['context','explicit_citation', 'implicit_citation_0.9', "human_labeled"]]
dataframe = dataframe.rename(columns={ 'explicit_citation': 'question', 'implicit_citation_0.9':'answer_text'})


cc_hl_df = dataframe[dataframe['human_labeled'] == 1]  # Replace 'column_name' with the name of the column you want to filter ondf.to_json('output.json', orient='records')  # Replace 'output.json' with the desired file name and path
cc_df = dataframe[dataframe['human_labeled'] == 0] 


# Assuming you have a dataframe called 'df'
df = cc_df.sample(frac = 1, random_state=42)


# Splitting data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Splitting the test set into test and validation sets
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

#Transfer from val to test
rows = val_df[:1000]
val_df[1000:]
test_df = pd.concat([test_df, rows])

#Add Human Labelled
val_df  = pd.concat([val_df, cc_hl_df])


# Printing the shapes of the datasets to verify the split
print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("Validation set shape:", val_df.shape)



def get_start_and_end(row):
    paragraph = row['context']
    context = row['answer_text']
    
    start_index = int(paragraph.find(context))
    end_index = int(start_index + len(context) - 1)
    return pd.Series([start_index, end_index])
    
train_df[['start_index', 'end_index']] = train_df.apply(get_start_and_end, axis=1)
test_df[['start_index', 'end_index']] = test_df.apply(get_start_and_end, axis=1)


model = RobertaForQuestionAnswering.from_pretrained("roberta-large")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")


max_length = 512
doc_stride = 128
pad_on_right = tokenizer.padding_side == "right"


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def convert_answers(r):
    start = r[0]
    text = r[1]
    return {
        'answer_start': [start],
        'text': [text]
    }

train_df['answers'] = train_df[['start_index', 'answer_text']].apply(convert_answers, axis=1)
test_df['answers'] = test_df[['start_index', 'answer_text']].apply(convert_answers, axis=1)
#df_train = dataframe[:-64].reset_index(drop=True)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenized_train_ds = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_test_ds = test_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

# ##### Define Hyperparameters

output_dir = "qa_model"
learning_rate = 2e-5
train_batch_size = 16
eval_batch_size = 16
num_train_epochs = 3
weight_decay = 0.01
data_collator = DefaultDataCollator()


# Define data loaders
train_dataloader = DataLoader(tokenized_train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_test_ds, batch_size=eval_batch_size , collate_fn=data_collator)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define scheduler
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,  # You can adjust this value
    num_training_steps=len(train_dataloader) * num_train_epochs
)


# Training Loop
for epoch in range(num_train_epochs):
    total_loss = 0.0
    
    # Training
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} (Training)")
    for step, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        total_loss += loss.item()
        progress_bar.set_postfix({'batch_loss': loss.item(), 'total_loss': total_loss / (step + 1)})

    print(f"Epoch {epoch + 1} Training Loss: {total_loss}")

    # Evaluation
    model.eval()
    total_eval_loss = 0.0
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    print(f"Epoch {epoch + 1} Evaluation Loss: {avg_eval_loss}")

# Save model if needed
model.save_pretrained(output_dir)

