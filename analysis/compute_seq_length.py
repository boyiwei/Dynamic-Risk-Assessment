from transformers import AutoTokenizer
import json

def compute_seq_length(dataset_path, tokenizer):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]
    
    max_len = 0
    max_i = 0
    len_conv = []
    filtered_data = []

    for i in range(len(data)):
        conv = data[i]['messages']
        tokenizer_conv = tokenizer.apply_chat_template(conv)
        conv_len = len(tokenizer_conv)
        
        if conv_len <= 20000:
            filtered_data.append(data[i])
            len_conv.append(conv_len)
        if conv_len > max_len:
            max_len = conv_len
            max_i = i

    len_conv.sort()
    print(max_i)
    print(len(len_conv))
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))  # Create larger figure
    plt.rcParams.update({'font.size': 14})  # Increase base font size
    
    plt.hist(len_conv, bins=50, edgecolor='black')
    # plt.title('Length Distribution of Conversations')
    plt.xlabel('Length (Number of Tokens)', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()  # Adjust layout to make room for larger fonts
    plt.savefig('length_distribution.pdf')
    plt.close()
    
    # Save the filtered data to a new file
    with open('low_confidence_intercode_ctf_Qwen2_single-turn_filtered.jsonl', 'w', encoding='utf-8') as file:
        for item in filtered_data:
            file.write(json.dumps(item) + '\n')
    
    return max_len

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    dataset_path = "analysis/successful_tasks_intercode_ctf_Qwen2_single-turn_subsettrain.jsonl"
    max_len = compute_seq_length(dataset_path, tokenizer)
    print(max_len)