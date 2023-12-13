# Fine-Tuning LLMs for Medical Entity Extraction
> #### _Archit | Fall '23 | Duke AIPI 591 (Independent Study in GenAI) Research Project_
&nbsp;  

## Project Overview ⭐

The pharmaceutical industry heavily relies on accurately processing adverse event reports, a task traditionally done manually and prone to inefficiencies. The automation of medical entity extraction, particularly `drug names` and `side effects`, is essential for enhancing patient safety and compliance. Current systems, while adept at identifying general information like patient name, age, address and other demographics, struggle with specialized medical terminology, creating a pressing need for more advanced solutions.

The goal of this project was to explore and implement methods for `fine-tuning` Large Language Models `(LLMs)` such as Llama2 and StableLM for the specific task of extracting medical entities from adverse event report (emails only). By fine-tuning these models on a synthetic dataset, derived from top drug information on Drugs.com, the aim was to surpass traditional entity recognition methods in accuracy and efficiency. This approach aims to streamline the entity extraction process and enhance the reliability and timeliness of data on drug adverse events, thereby offering potential improvements in medical data analysis practices.

&nbsp;
## Data Sources 💾 

This project's dataset was built by extracting detailed information about the top 50 most popular drugs from [Drugs.com](https://www.drugs.com), a comprehensive and authoritative online resource for medication information. Drugs.com provides a wide range of data on pharmaceuticals, including drug descriptions, dosages, indications, and primary side effects. This rich source of information was instrumental in developing a robust and accurate synthetic dataset for the project.

The drugs selected for this study include a wide range of medications known for their prevalence in the market and significance in treatment regimens. These drugs span various therapeutic categories and include:

* `Psychiatric` drugs like Abilify.
* `Immunomodulators` such as Infliximab, Rituximab, Etanercept, Humira, and Enbrel.
* `Gastrointestinal` medications like Nexium, Prevacid, Prilosec, and Protonix.
* `Cholesterol`-lowering agents including Crestor, Lipitor, Zocor, and Vytorin.
* `Diabetes` medications such as Victoza, Byetta, Januvia, and Onglyza.
* `Respiratory` treatments like Advair, Symbicort, Spiriva, and Singulair.
* `Erectile dysfunction` drugs including Cialis, Viagra, Levitra, and Staxyn.
* `Other Medications` like AndroGel, Prezista, Doxycycline, Cymbalta, Neupogen, Epogen, Aranesp, Neulasta, Lunesta, Ambien, Provigil, Nuvigil, Metoprolol, Lisinopril, Amlodipine, Atorvastatin, Zoloft, Lexapro, Prozac, Celexa, and Atripla.

Each of these drugs was carefully chosen to provide a comprehensive view of the different types of medical entities that the LLMs would need to identify and extract from adverse event reports.  

&nbsp;
## Data Processing 📝  

### **Scraping drug information data from the Drugs.com**

For this project, crucial drug information was scraped from Drugs.com. Each drug's dedicated webpage provides detailed information which varies in structure, making the scraping process complex.

To effectively handle this complexity, a Python script utilizing the [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) library was employed. This script parsed the HTML content of each webpage, targeting specific sections relevant to our study, such as drug uses and side effects. For text conversion, the [html2text](https://pypi.org/project/html2text/) package was used, allowing the extraction of clean and readable text data from the HTML content.

The python script  to scrape text can be found in the `scripts` folder and can be run as follows:

**1. Create a new conda environment and activate it:** 
```
conda create --name llms python=3.10
conda activate llms
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```
**3. Run the web scraping script:** 
```
python scripts/scrape_drugs_data.py
```  

&nbsp;    
### **Synthetic Dataset Generation for Fine-Tuning**

Using the scraped drug information, synthetic `adverse event report emails` were generated. These emails simulate real-world data while ensuring that no real patient data or personally identifiable information was used. The generation process was carried out using prompts designed to guide `ChatGPT` in creating realistic and relevant data scenarios for training purposes. The prompt template used can be found in the data folder and is as follows:  
```
Act as an expert Analyst with 20+ years of experience in Pharma and Healthcare industry. You have to generate Adverse Event Reports in JSON format just like the following example:

{
    "input": "Nicole Moore
            moore123nicole@hotmail.com
            32 McMurray Court, Columbia, SC 41250
            1840105113, United States 
            
            Relationship to XYZ Pharma Inc.: Patient or Caregiver
            Reason for contacting: Adverse Event
            
            Message: Yes, I have been taking Metroprolol for two years now and with no problem. I recently had my prescription refilled with the same Metoprolol and I’m having a hard time sleeping at night along with running nose. Did you possibly change something with the pill...possibly different fillers? The pharmacist at CVS didn’t have any information for me. Thank you, Nicole Moore", 
    "output": { 
                "drug_name":"Metroprolol", 
                "adverse_events": ["hard time sleeping at night", "running nose"]
            }
}

Now create Adverse Event Reports in a similar way for the Drug - ''' [DRUG NAME] '''

You have more information about the drug's use and its side effects below - ''' [DRUG SIDE EFFECTS] '''
```

The synthetic training dataset was generated with groundtruth "input" and "output" pairs, preparing them for use in fine-tuning the language models. This included labeling the relevant entities specifically `drug_name` and `adverse_events`. 

&nbsp;  
Following is an example of the generated data:  
```
{
    "input": "Natalie Cooper,\nncooper@example.com\n6789 Birch Street, Denver, CO 80203,\n102-555-6543, United States\n\nRelationship to XYZ Pharma Inc.: Patient\nReason for contacting: Adverse Event\n\nMessage: Hi, after starting Abilify for bipolar disorder, I've noticed that I am experiencing nausea and vomiting. Are these typical reactions? Best, Natalie Cooper",
    
    "output": "{\"drug_name\": \"Abilify\", \"adverse_events\": [\"nausea\", \"vomiting\"]}"
}
```  
The python script to generate synthetic data can be found in the `scripts` folder. Assuming you are in the same conda environment as the previous step, the python script can be run as follows:

**1. Run the data generation for preparing dataset:** 
```
python scripts/data-prepare.py 
```
**2. Run the data aggrgatation script to prepare train and test splits:** 
```
python scripts/combine-data.py 
```

These scripts will generate a supervised dataset with `input` and `output` pairs where input is the adverse event email and the output is the extracted entities. The generated data its stored in  `entity-extraction-train-data.json` and `entity-extraction-test-data.json` files in the `data/entity-extraction` folder. We have 700 training samples and ~70 test samples.  

&nbsp;  
## Fine-tuning Large Language Models (LLMs) for Medical Entity Extraction 🧠    

In this project, two Large Language Models (LLMs), `Llama2` and `StableLM`, were fine-tuned using techniques such as `Parameter Efficient Fine-Tuning (PEFT)`, specifically through `Adapter V2` and `LoRA (Low-Rank Adaptation)` methods. PEFT techniques allow for the modification of large models without having to retrain all the parameters, making the fine-tuning process more efficient and resource-friendly. This approach is particularly valuable for tasks that require domain-specific adaptations without losing the broad contextual knowledge the models already possess.

The fine-tuning assess and compares the effectiveness in enhancing the models' performance for medical entity extraction. These approaches were aimed to balance efficiency and precision, ensuring that the models could accurately identify and extract relevant medical information from complex textual data.  

### **Approach 1: Fine-tuning using LoRA Parameter Efficient Fine-Tuning (PEFT)**

LoRA focuses on updating the weight matrices of the pre-trained model through low-rank matrix decomposition. By altering only a small subset of the model's weights, LoRA achieves fine-tuning with minimal updates, maintaining the model's overall structure and pre-trained knowledge while adapting it to specific tasks.

In this approach, only a limited set of weights are fine-tuned on the Synthetic Medical Entity Dataset we generated. The hyperparameters used are as follows:
- **Model:** Llama-2-7b or stable-lm-3b
- **Batch Size:** 16
- **Learning Rate:** 3e-4
- **Weight Decay:** 0.01
- **Epoch Size:** 700
- **Num Epochs:** 5
- **Warmup Steps:** 100

The data is first prepared by tokenizing the text data and converting it into a torch dataset. The model is then fine-tuned on the data using the [Lightning](https://www.pytorchlightning.ai/) framework.

The model is fine-tuned on a 1 GPU (48GB) for 5 epochs. The data preparation and fine-tuning scripts can be found in the `scripts` and `finetune` folders respectively. Assuming you are in the same conda environment as the previous step, the python script can be run as follows:

**1. Prepare data for fine-tuning (Stable-LM):** 
```
python scripts/prepare_entity_extraction_data.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```
**2. Run the fine-tuning script (Stable-LM) :** 
```
python finetune/lora.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b --out_dir out/lora/Stable-LM/entity_extraction
```
**3. Prepare data for fine-tuning (Llama-2):** 
```
python scripts/prepare_entity_extraction_data.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```
**4. Run the fine-tuning script (Llama-2) :** 
```
python finetune/lora.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --out_dir out/lora/Llama-2/entity_extraction
```

### **Approach 2: Fine-tuning using Adapter Parameter Efficient Fine-Tuning (PEFT)**

The Adapter-V2 technique involves inserting small, trainable layers (adapters) into the model's architecture. These adapters learn task-specific features while the majority of the model's original parameters remain frozen. This approach enables efficient fine-tuning, as only a small fraction of the model's parameters are updated, reducing computational requirements and preserving the pre-trained knowledge.

In this approach, only a limited set of weights are fine-tuned on the Synthetic Medical Entity Dataset we generated. The hyperparameters used are as follows:
- **Model:** Llama-2-7b or stable-lm-3b
- **Batch Size:** 8
- **Learning Rate:** 3e-3
- **Weight Decay:** 0.02
- **Epoch Size:** 700
- **Num Epochs:** 5
- **Warmup Steps:** 2 * (epoch_size // micro_batch_size) // devices // gradient_accumulation_iters

The data is first prepared by tokenizing the text data and converting it into a torch dataset. The model is then fine-tuned on the data using the [Lightning](https://www.pytorchlightning.ai/) framework.

The model is fine-tuned on a 1 GPU (24GB) for 5 epochs. The data preparation and fine-tuning scripts can be found in the `scripts` and `finetune` folders respectively. Assuming you are in the same conda environment as the previous step, the python script can be run as follows:

**1. Prepare data for fine-tuning (Stable-LM):** 
```
python scripts/prepare_entity_extraction_data.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```
**2. Run the fine-tuning script (Stable-LM) :** 
```
python finetune/adapter_v2.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b --out_dir out/adapter/Stable-LM/entity_extraction
```
**3. Prepare data for fine-tuning (Llama-2):** 
```
python scripts/prepare_entity_extraction_data.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```
**4. Run the fine-tuning script (Llama-2) :** 
```
python finetune/adapter_v2.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --out_dir out/adapter/Llama-2/entity_extraction
```  

&nbsp;  
## Performance Evaluation and Metrics 📊  

The effectiveness of the fine-tuned models was evaluated based on their precision and recall in identifying medical entities. These metrics provided insights into the models' accuracy and reliability compared to each other when trained using different techniques and to their base versions.

The test dataset which was kept aside during the data generation process was used to evaluate the performance of the fine-tuned models. The test dataset contains 70 samples and the performance of the models was evaluated on precision and recall for the `drug_name` and `adverse_events` entities.

Based on the evaluation, the following table shows the performance of the different models:

| Model Type | Training Technique | Precision | Recall |
| --- | :---: | :---: | :---: |
| Llama-2-7b-hf | Base Model | 100% | 0% |
| Llama-2-7b-hf | PEFT (LoRA) | 20% | 40% |
| Llama-2-7b-hf | PEFT (Adapter) | 20% | 40% |
| stablelm-base-alpha-3b  | Base Model | 100% | 0% |
| stablelm-base-alpha-3b  | PEFT (LoRA) | 20% | 40% |
| stablelm-base-alpha-3b  | PEFT (Adapter) | 20% | 40% |  

&nbsp;  
## Future Work 📈  

Potential future developments include creating a user-friendly interface or tool that leverages these fine-tuned models. Such a tool would be invaluable for pharmaceutical companies and medical professionals, enabling efficient and accurate extraction of medical entities from various reports. We also plan to include more variety in the data, including more drugs and more side effects. We can also look into pre-training the LLMs on a larger biomedical dataset and then fine-tuning them on the real world medical adverse event reports. This will help the model learn more about the different types of medical entities and improve its performance.  

&nbsp;  
## Project Structure 🧬  
The project structure is as follows:
```
├── data                                        <- directory for project data
    ├── entity-extraction                       <- directory for processed entity extraction data
        ├── entity-extraction-data.json         <- full prepared synthetic dataset    
        ├── entity-extraction-test-data.json    <- test data for entity extraction
        ├── entity-extraction-train-data.json   <- train data for entity extraction
        ├── train.pt                            <- python pickle file for train data
        ├── test.pt                             <- python pickle file for test data
    ├── entity_extraction_reports               <- directory of generated adverse event reports
        ├── [DRUG NAME].json                    <- synthetic adverse event report for the drug
    ├── raw_drug_info                           <- directory for raw scraped drug information
        ├── [DRUG NAME].txt                     <- raw scraped drug information
    ├── predictions-llama2-adapter.json         <- predictions of the Llama-2 fine-tuned using Adapter PEFT
    ├── predictions-llama2-lora.json            <- predictions of the Llama-2 fine-tuned using LoRA PEFT
    ├── predictions-stablelm-adapter.json       <- predictions of the Stable-LM fine-tuned using Adapter PEFT
    ├── predictions-stablelm-lora.json          <- predictions of the Stable-LM fine-tuned using LoRA PEFT
    ├── prompt-template.txt                     <- prompt used to generate synthetic data
├── finetune                                    <- directory for fine-tuning scripts
    ├── adapter_v2.py                           <- script to fine-tune LLMs using Adapter PEFT
    ├── lora.py                                 <- script to fine-tune LLMs using LoRA PEFT
├── generate                                    <- directory for inference scripts
    ├── base.py                                 <- script to generate predictions using base LLMs
    ├── adapter_v2.py                           <- script to generate predictions using Adapter PEFT
    ├── lora.py                                 <- script to generate predictions using LoRA PEFT
├── lit_gpt                                     <- directory for LIT-GPT Framework code
├── notebooks                                   <- directory to store any exploration notebooks used
├── performance_testing                         <- directory to store performance testing data
    ├── test_answers_analysis.xlsx              <- analysis of the answers returned by the pipeline
├── scripts                                     <- directory for pipeline scripts or utility scripts
    ├── combine_data.py                         <- script to combine the generated data for all drugs
    ├── convert_hf_checkpoint.py                <- script to convert a HuggingFace checkpoint to a LIT-GPT checkpoint
    ├── data-prepare.py                         <- script to prepare the synthetic dataset for fine-tuning
    ├── download.py                             <- script to download the pre-trained LLMs from HuggingFace
    ├── evaluate.py                             <- script to evaluate the performance of the fine-tuned models
    ├── prepare_entity_extraction_data.py       <- script to tokenize the processed data and create torch datasets
    ├── scrape_drugs_data.py                    <- script to scrape drug information from Drugs.com
├── .gitignore                                  <- git ignore file
├── LICENSE                                     <- license file
├── README.md                                   <- description of project and how to set up and run it
├── requirements.txt                            <- requirements file to document dependencies
```

## References 📚

- [Lightning-AI/lit-gpt](https://github.com/Lightning-AI/lit-gpt/tree/cf5542a166d71c0026b35428113092eb41029a8f)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [ChatGPT API](https://platform.openai.com/docs/guides/chat)