# Technical Report

**Project: Analyzing hatefulness and toxicity in Reddit Discussions**  

Members: Lim Jia Xuan Eileen, Lynn Keok, Tan Sze Jing, Yeo Hiong Wu

Last updated on: 12 November 2024

## Section 1: Context

Policymakers and agencies have become increasingly aware of the shift towards more polarised conversations on social media platforms. As a platform that encourages free discussions by allowing users to express their thoughts and opinions anonymously, Reddit has seen a rise in open discussion and toxic discourse. This led to a growing interest in understanding whether online conversations have become more hostile and negative over recent years and why this trend is happening. 

## Section 2: Scope

### 2.1 Problem

The MDDI’s Online Trust and Safety department faces a challenge of managing and understanding the increasingly hateful and toxic comments on Internet platforms. With users reflecting the rise in harmful online content compared to the past year, it highlights the concerning trend towards more hostile online discussion. This poses a threat to maintaining a safe online environment, especially for vulnerable groups like youth and children. With partnerships with platforms like Tiktok, Google and Meta, a study to analyse the rise in hateful and toxic discussions is necessary to determine policy decisions and intervention strategies efficiently.


Failure to resolve the problem of mitigating hateful and toxic comments online can contribute to a more polarised society, especially for a multi-religious and multi-racial country like Singapore. The increase in harmful content over the past year increases exposure of negative influences to youth and children. Without timely and proper intervention, the growing hostile discussion threatens community cohesion by amplifying negative stereotypes and widening divides among social groups. 


Given the volume of comments generated on online platforms, manually reviewing and moderating content is not feasible. Hence, data science is required due to its ability to process and analyse vast amounts of data accurately and efficiently.  Machine learning enables scalable and automated analysis, which helps to ensure timely interventions and maintain a safe online environment for users. 


### 2.2 Success Criteria

Goal 1: Success will be achieved if the analysis yields actionable insights into specific trends, such as identifying the primary topics that generate hatefulness or periods of heightened toxicity, thereby informing targeted policy decisions. 


Goal 2: Success will be achieved if the system can handle and process the data within a reasonable time (eg. processing one year of comments within a day) and generate timely, interpretable reports on toxicity trends. Furthermore, should the analysis framework be adaptable to analyse data from other social platforms (eg. Twitter, Facebook) or extendable to future Reddit data, it would ensure support for ongoing and expanded toxicity analysis efforts and accommodate a growing volume and variety of social media data as needed.


### 2.3 Assumptions

This project assumes that there will be sufficient computational resources such as high RAMS, GPUs and adequate computer units to handle the large datasets efficiently. Limitations in these resources can delay the model processing and execution. 


This project assumes the use of models from Hugging Face for sentiment analysis, hatefulness and toxicity detection due to time and resource constraints. Output generated from these models is assumed to provide reasonably accurate predictions for the classification of comments found on Singapore reddits. 


The choice of models from HuggingFace is assumed to generalise well to comments on Singapore reddit. If the models’ predictions are not sufficiently accurate, manual labelling may become necessary, which could introduce bias and increase processing time. Further fine-tuning may be required, which can affect the project timeline and quality of insights.


## Section 3: Methodology

### 3.1 Technical Assumptions

The following key terms are defined as follows:

**Hate**: Language that targets individuals or groups based on characteristics such as race, gender, religion, or nationality, often with harmful or demeaning intent. It may incite violence or promote hatred against specific groups.

**Toxic**: Language that is harmful, abusive, or offensive, potentially including insults, threats, and derogatory remarks.

Hate can be seen as a subset of toxicity. It can be viewed as a type of toxic language that targets groups or individuals based on identity while toxic language is a broader category that includes any harmful or offensive speech, not necessarily directed at specific groups.

The datasets contain several features that provide information for analysis. The text column contains the raw content of Reddit comments, while the timestamp records the time each comment was posted. The username feature represents the Reddit user's identifier. Link, link_id, and parent_id serve as identifiers to track the structure of comment threads. Each comment is assigned a unique ID, and the subreddit_id indicates the specific subreddit where the comment was posted. However, the datasets lack true labels for sentiment, hatefulness, or toxicity, which will be predicted through models. User profile information, such as engagement level or user history, is also not available in the datasets. 

This project is conducted primarily on Google Colab, with the option to upgrade to paid GPU services to address resource limitations. The paid services provide access to advanced GPUs, such as the T4 and A100, along with high-memory configurations of up to 51 GB RAM. It is also assumed that sufficient OpenAI API credits have been purchased for use in the topic modelling process to generate key topics driving toxicity and hatred.

The key hypotheses of this project are:

1. There has been an increase in negative sentiment in Singapore subreddits from 2020 to 2023.
2. The prevalence of hateful and toxic language in Singapore subreddits has increased from 2020 to 2023.
3. The increase in hateful and toxic language during specific months are driven by the emergence of certain events that may invoke strong reactions and discussions. 



### 3.2 Data

The data consists of two distinct datasets of Singapore Reddit comments:

1. Reddit Threads from 2020 to 2021: Comments sourced from the subreddits r/Singapore and r/SingaporeRaw.

2. Reddit Threads from 2022 to October 2023: Comments sourced from r/Singapore, r/SingaporeRaw, and r/SingaporeHappenings.

Preprocessing steps were applied to the text column in the datasets. Comments marked as [deleted] or [removed] were excluded. Special characters, including emojis, were removed and contractions were expanded to their full forms to standardise the text. All text was converted to lowercase to ensure uniformity. Single-letter words such as A and I were removed, as they do not contribute meaningfully to the analysis. Entries with empty text were also dropped. However, comments that were made by usernames that were marked as [deleted] were kept as we assume that they would tend to be more toxic in nature, to accurately capture the extent of toxicity.

Two new columns year and month were created to indicate when each comment was posted. A new column named subreddit_name was added to identify the subreddit label (r/Singapore, r/SingaporeRaw, and r/SingaporeHappenings) based on the subreddit_id present in the dataset. 

Analysis is conducted on a 20% sampled subset of each dataset to improve efficiency. Stratified sampling was applied based on the subreddit labels to ensure each subreddit’s comments are proportionately included as r/Singapore, with its large member base of 1.5 million, is heavily represented compared to r/SingaporeRaw (77,000 members) and r/SingaporeHappenings (43,000 members). The samples also retain the same yearly comment percentages as the original datasets, allowing for reliable insights without needing to process the entire data.

Tokenization and embedding were conducted using the SingBERT model from Hugging Face, which has been fine-tuned on Singlish and Manglish data retrieved from Reddit threads in r/Singapore and r/Malaysia. The sentence-level embeddings were generated using the pooler output layer with the [CLS] token. Tokens and token IDs were retrieved using the same model.

This allowed us to create 4 files which can be used for analysis: 

1. sample_2021.csv: sample subset containing 20% of Reddit threads from 2020 to 2021. It includes preprocessed text columns,  year, month, subreddit name, token IDs, and tokens.
2. sample_2223.csv: sample subset containing 20% of Reddit threads from 2022 to October 2023. It includes preprocessed text columns,  year, month, subreddit name, token IDs, and tokens.
3. embeddings_2021.npy: contains the embedding output for the text column from sample_2021 data
4. embeddings_2223.npy: contains the embedding output for the text column from sample_2223 data



### 3.3 Experimental Design

**Sentiment Analysis**

The Twitter RoBERTa Base Sentiment model was chosen for sentiment analysis because of its specialisation in understanding sentiment expressed in social media text, such as Twitter data, which is similar to Reddit data.  This model, based on the architecture, can provide strong performance for short-form, informal language often found in social media, capturing nuanced sentiments that more generalised models may miss. The model is also optimised for classification accuracy across various sentiments, making it well-suited to capture complex sentiment patterns in real-world text data. 

To process the data, we implemented a batch-based approach for efficient computation. The sentiment analysis pipeline involved the following steps:

1. **Tokenization with Truncation**: We tokenized each text using a maximum sequence length of 512 tokens. This truncation method ensures that any text longer than 512 tokens is truncated, capturing the most relevant parts of each text while preventing excessive memory usage.
2. **Batch Processing**: We divided the data into batches of 128 texts, allowing the model to handle large datasets efficiently and leverage GPU capabilities when available.
3. **Model Inference**: For each batch, we ran a forward pass through the Twitter RoBERTa Base model, obtaining logits for each text. These logits were then passed through a softmax layer to calculate class probabilities.
4. **Sentiment Prediction and Confidence Scoring**: We used the highest probability to assign a sentiment label ("negative," "neutral," or "positive") to each text and recorded the model's confidence in its prediction.
Before deciding on using Twitter RoBERTa Base model, we tried using DistilBERT. While DistilBERT is efficient and lightweight, it sacrifices some depth in language understanding for social media-specific text and may not capture the subtleties of sentiment as accurately. Hence, we decided on using Twitter RoBERTa Base Sentiment model. 


**Hate and Toxicity Analysis**

Several Hugging Face models were used to detect hateful and toxic content. Using multiple models leverages the strength of each model while reducing individual biases. This helps capture diverse nuances in hateful and toxic language, leading to more reliable and comprehensive classification.

The Twitter RoBERTa Base Hate Classifier model was used to detect hateful content because it is fine-tuned for social media environments like Twitter, which share similar dynamics as Reddit, such as open discussions and informal language. The model was fine-tuned on 13 hate speech datasets from various platforms, including data targeting specific groups based on attributes such as race, gender, and nationality. This allows the model to recognize a wide range of hateful expressions in different contexts. Each text entry in our samples is processed through this model, receiving a 'HATE' label if classified as hateful or 'NON-HATE' otherwise.

The RoBERTa Toxicity Classifier model was chosen due to its fine-tuning on toxicity detection, making it suited for identifying abusive language and offensive speech in various online contexts. This model was trained on datasets from Civil Comments and Wikipedia talk page. Civil Comments uses a crowd-sourced moderation system that allows users to rate each other’s comments on civility and toxicity while wikipedia talk pages are collaborative spaces where discussions and debates occur, sometimes leading to contentious interactions. This enables it to capture various forms of toxicity in online discussions, including insults, threats, and harmful expressions. Each text entry in our samples is processed through this model, receiving a 'toxic' label if classified as toxic or 'neutral' otherwise.

Both models above share the RoBERTa architecture, which performs well for many NLP tasks. RoBERTa was trained on larger data and bigger batch sizes which allows it to recognize complex patterns in text more efficiently, making it useful for identifying hateful and toxic content where contexts and nuances are important. Similar to the sentiment analysis model, tokenization with truncation and batch processing were utilised to make the process more efficient given the large data size. Both RoBERTa model tokenises the input text and uses it to create embeddings. A classification layer is then applied to predict the hate and toxicity level. 

The Detoxify model was trained on the same datasets as the RoBERTa Toxicity Classifier model, but was trained specifically to predict toxic comments on 3 Jigsaw challenges: Toxic comment classification, Unintended Bias in Toxic comments, and Multilingual toxic comment classification. From the 3 models provided by Detoxify, we chose to utilise the unbiased model in an effort to minimise unintended bias with respect to mentions of identities, as it was discovered that toxicity models learned to associate the names of frequently attacked identities with toxicity. Given Singapore’s unique racial and religious dynamics, we concluded that it was crucial to account for such unintended bias. The unbiased model predicts the probability for 7 different labels: ‘toxicity’, ‘severe_toxicity’, ‘obscene’, ‘threat’, ‘insult’, ‘identity_attack’, and ‘sexual_explicit’. In congruence with our definitions of hate and toxicity, we only focused on the ‘toxicity’ and ‘identity_attack’ labels and set a threshold of 0.5, such that a probability larger than or equal to 0.5 would be assigned 1, a positive label, and a probability smaller than 0.5 would be assigned 0, a negative label. 


**Topic Modelling**

Additional Preprocessing:

Taking the union of hateful and toxic text identified from the three models enables us to create datasets that capture all instances flagged as either hateful or toxic. The NLTK stopwords library was used to remove common stopwords and punctuation. Frequently used words in the Singapore context, such as 'lah', 'leh', and 'lor' were also excluded. This ensures that the topic modelling focuses on more meaningful and relevant terms. 

Two additional datasets were created for topic modelling: 

1. hatetoxic_2021.csv: contains all text identified as hateful or toxic in 2020 and 2021, with the additional column of text with stopwords and punctuations removed
2. hatetoxic_2223.csv: contains all text identified as hateful or toxic in 2022 and 2023, with the additional column of text with stopwords and punctuations removed


Algorithm Selection:

1. First Stage - Keyword Clustering

First, zero-shot classification was considered. It is a technique which requires topics to be known beforehand, so that texts can be classified directly into any of the topics and the top topics of the month causing toxicity and hate can be found. However, it was decided that this method was too deterministic and did not really give us any real insights into the true causes of toxicity and hate. The topics had to be formed by the texts themselves and not inserted by us or experts with domain knowledge. Thus, other methods of topic modelling were explored.

Then, LDA was used, a popular probabilistic topic modelling technique. LDA is able to group similar words together and find common topics based on co-occurrence. However, LDA struggled to generate meaningful topics. This is likely due to LDA working better on longer texts where it can observe a wide range of word patterns, and that the data is in the form of short, unstructured texts, often containing many generic words and few words that actually point to a topic. This leads to common words being labelled as topics, generating nonsensical results that do not give any insights. The approach was then modified, with the model asked to instead cluster words freely and return the top 10 keywords of each cluster. Each cluster would then represent a distinct topic. Similarly, the keywords returned were all common words. The hyperparameter of cluster number was changed multiple times, all to the same result. It is also noted that many of the keywords were vulgarities. This is understandable as toxic and hateful texts would likely contain many vulgarities. A custom stopword list consisting of many observed vulgarities was included as at this stage of topic modelling, the vulgarities would not yield any meaningful topic and would instead form a nonsensical topic cluster of their own. The results afterwards were slightly improved, but still insufficiently focused. It was concluded that this was a result of how the model worked, and that it was more targeted towards longer documents. Thus, another model was sought.

Given the limitations of LDA, BERTopic was used instead. It leverages BERT embeddings for improved topic modelling. It required much longer processing times than LDA (approximately 5 times the amount of time). It did not do very well when asked to output a single word to represent each topic cluster. Therefore, a similar approach was used where the model would return the top 10 keywords of each topic cluster, and it was able to freely decide the number of clusters returned. It did generate much better topic clusters. This is likely due to BERTopic's ability to use contextual word embeddings, clustering them into groups that represent topics, rather than relying on co-occurrence patterns. It is thus much better suited towards understanding short texts with limited context, such as Reddit comments. BERTopic models are also able to automatically determine the optimal number of topic clusters to be output based on the data. This resulted in better topic clustering and identification, with several of the topic clusters quite well defined through closely associated keywords. The common topic can easily be identified by finding the commonality in the keywords.

Some of the hyperparameters tuned:

* Sentence Transformer Model "all-mpnet-base-v2" was used as it outperforms other models like BERT on semantic textual similarity tasks, which can lead to better topic separation in BERTopic. It also is known for generating high-quality sentence embeddings, capturing nuanced semantic relationships between words, which is particularly beneficial when working with social media text, where language may be informal and context-dependent. However, using this leads to longer computational times due to lack of GPU, which would be a limitation when working with the full dataset.
* A low cluster_selection_epsilon value of 0.01 for HDBSCAN	 was used as it increases the distinctions between clusters, potentially separating overlapping clusters, leading to more interpretable topics. However, it leads to longer computational times.
* Setting nr_topics="auto" enables BERTopic to automatically merge similar topics, reducing noise and focusing on the most prominent topics and themes in the dataset.


2. Second Stage - Topic Formation

The next step in the process was to find a representative word or phrase to represent the topic cluster. This word or phrase will capture the essence of all the keywords, and will be a common point that all the keywords of that cluster are associated with.

Firstly, Hugging Face sentence transformers were used to attempt this. However, it was ill-suited to the task as it is designed to find semantic similarity between longer texts rather than shorter texts or even words, as in this case. It always picked one word out of the 10 keywords to be the representative word. This could be useful in some cases. For example, if the words were [government, hospital, grassroots, school] and the model picked government, it would be a decent representation of the topic of the cluster. However, the converse is also true; if the model picked school instead, it would be a bad representation of the topic of this cluster. The issue likely stems from sentence transformers being optimised for sentences and not isolated keywords. Singular words lack key context and semantic information, the model is unable to infer a clear topic and treats each word as a distinct concept. It therefore just picks any one of the keywords. The problem therefore is that the model is not generalising from the keywords of the topic cluster.

It was discovered that ChatGPT was excellent at generalising and creating useful representative words or phrases to use as topics. GPT models are highly effective at identifying common themes from a set of keywords due to their advanced capabilities in contextual understanding and natural language processing. Their diverse training allows them to recognize semantic associations and relationships between words, even in cases where context is sparse. Unlike models that rely on explicit sentence structure such as Hugging Face sentence transformers, GPT can infer meaning from isolated keywords. It treats the keywords as related rather than isolated. Its ability to perform zero-shot learning enables it to understand tasks it has not been explicitly trained for, making it highly adaptable for tasks like identifying topics from a set of keywords. OpenAI’s API was therefore used to this end. It was extremely successful at finding common topics amongst the keywords. 

It was then fine-tuned by being pointed towards a Singapore context so that abbreviations could be understood from a Singaporean point of view, and also directed to generate topics that might be points of contention in an online forum.


3. Third Stage - Topic Classification

To find the most popular topics of the month driving toxicity and hate, Zero-Shot classification was utilised to efficiently categorise the large volume of text data into predefined topics identified from the previous steps. Zero-Shot classification is particularly valuable in this context because it allows for accurate text classification without the need for extensively labelled training data, which would be time-consuming and costly to generate. By leveraging a model's ability to attribute a given text to a known set of topics, zero-shot classification can assign texts to their most relevant topic even when the topics were not explicitly seen during training. 

This approach ensures that we can quickly assess the prevalence of each topic within the dataset through obtaining counts of the number of texts that evaluate to each topic. Comparing against the number of toxic and hateful texts per month, it is then easy to tell what are the main topics driving hate and toxicity through proportion. Zero-Shot classification therefore provides a scalable and flexible solution for understanding the underlying patterns in toxic comments without manual labelling or retraining the model each time the topics change.

Epsilon value was the hyperparameter tuned for this part of the model. It was first observed that the similarity values were quite low, mostly hovering at below 0.2. Recall was prioritised in this situation as we did not mind if there were false positives, but were more concerned about getting the texts mostly classified into the generated topics. Thus, the threshold was set quite low at 0.1. Any text with similarity to all generated topics below 0.1 were classified as ‘others’ as its topic, as it was not similar to any of the main topics generated. 

Afterwards, automated epsilon value hyperparameter tuning was attempted, using f1 score as the metric for evaluation as it balances both precision and recall. In this task, we are trying to ensure that most texts are evaluated to a generated topic rather than ‘others’ through a low epsilon value, while not compromising too much on precision. Therefore, there needs to be a certain balance struck between precision and recall, thus f1 score was used as the metric to automate the process of choosing the optimal epsilon, which resulted in the same epsilon of 0.1 being the optimal.


Additional notes:

It was always the goal to do a monthly topical analysis as a key aim of the topic modelling section was to find out the key topics driving toxicity and hatred on Singaporean subreddits. The splits were therefore done by year, then by month. 

Initially, there was consideration on whether there was to be a split by subreddit. However, on running the models on a subreddit split, it is discovered that the topics driving toxicity and hatred are not substantially different from one another over the 3 subreddits. Therefore, the analysis was done over all 3 subreddits together.

## Section 4: Findings

### 4.1 Results

**Sentiment Analysis**

After doing sentiment analysis on 20% of the sample data from 2020 to 2023, we realised that the sentiment results reveal a clear trend that negative sentiment has steadily increased each year, rising from 34.50% in 2020 to 42.21% in 2023. Conversely, positive sentiment showed a gradual decline, decreasing from 15.64% in 2020 to 11.10% in 2023. Neutral sentiment remained relatively stable, with a slight downward trend from 49.86% in 2020 to 46.69% in 2023. These results suggest an overall shift toward a more negative tone over the years, with a corresponding decrease in positive sentiment, while neutral sentiment levels maintained a relatively consistent proportion. 
![alt text](images/table1.png>)
![alt text](<images/graph1.png>)
![alt text](<images/graph2.png>)
![alt text](<Screenshot 2024-11-12 at 9.52.36 PM.png>)

**Hate and Toxicity Analysis**

Time series analysis on the combined hate and toxicity datasets also revealed similar results to sentiment analysis, where both toxicity and hate proportions show an increasing trend over the years, with local peaks in October 2021 and August 2022, although it can be noted that the proportion of hate and toxic comments dipped after August 2022.

![alt text](<Screenshot 2024-11-12 at 9.54.06 PM.png>)
![alt text](<Screenshot 2024-11-12 at 9.54.12 PM.png>)

Username analysis:

To find out if a subset of users contributed disproportionately to toxic discourse, the dataset was filtered to contain only toxic comments and the number of times a user commented in a toxic manner was counted. The total 111,118 toxic comments were made by just 18,892 users, giving us an average of 6 comments per user from 2020 to 2023. However, the vast majority of users fall below this average, with 3374 users having made 6 or more toxic comments. By ranking the counts of toxic comments per user, we found that the bulk of toxic comments were made by just a small group of users, with the top user “tom-slacker” having made 675 toxic comments. 

Time series analysis was also done for the top 3 commenters to observe their commenting patterns. The number of toxic comments made for all three users fluctuated significantly from month to month, reaching upwards of 30 to 40 toxic comments a month, or around 1 toxic comment a day. There would be periods of high activity and periods of inactivity, which could hint at certain events that encourage these users to return with toxicity. 

![alt text](<Screenshot 2024-11-12 at 9.54.26 PM.png>)
![alt text](<Screenshot 2024-11-12 at 9.54.34 PM.png>)
Interestingly enough, these three redditors have only posted toxic comments on the r/singapore subreddit.

Subreddit analysis:

![alt text](<Screenshot 2024-11-12 at 9.55.15 PM.png>)
Overall, the subreddit r/Singapore has nearly half the proportion of toxic comments as compared to r/SingaporeHappenings and r/SingaporeRaw. This could be attributed to the fact that r/Singapore has a much larger community and is much more general than the other two subreddits, whereas r/SingaporeRaw specifically provides a space for controversial conversations. The time series plot shows wild fluctuations in toxic comment proportion for r/SingaporeRaw throughout 2020, which could be attributed to the outbreak of COVID. In contrast, toxic comment proportions stayed relatively stable throughout for r/Singapore, further supporting the previous hypothesis. Lastly, while r/SingaporeHappenings was only created in September 2022, it has attracted even more toxic comments relative to r/SingaporeRaw in its first few months, possibly acting as a new breeding ground for toxicity. Since r/SingaporeHappenings is a subreddit that centres around daily life in Singapore, it is likely the place Singapore residents go to voice their complaints. The explosion of toxicity in its first month may be an indicator of the dissatisfaction Singapore residents have towards their quality of life.
![alt text](<Screenshot 2024-11-12 at 9.58.40 PM.png>)

**Topic Modelling**

On completion of topic modelling over the years of 2020 to 2023, several large trends can be observed regarding the topics driving toxicity and hatred on the forums on a monthly level.

Below are the results obtained through analysis by the model:

**2020**
![alt text](<Screenshot 2024-11-12 at 9.59.57 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.00.23 PM.png>)
**2021**
![alt text](<Screenshot 2024-11-12 at 10.02.36 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.02.44 PM.png>)
**2022**
![alt text](<Screenshot 2024-11-12 at 10.03.48 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.03.57 PM.png>)
**2023**
![alt text](<Screenshot 2024-11-12 at 10.04.04 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.04.25 PM.png>)

There are 3 immediate findings that can be noted from observing the results: the main topics driving toxicity and hate amongst Singaporeans, topics that cause spikes in toxicity and hate and notable topics.

Below are tables that contain the definitions of these terms and findings.
![alt text](<Screenshot 2024-11-12 at 10.05.57 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.07.13 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.08.06 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.08.32 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.08.50 PM.png>)
![alt text](<Screenshot 2024-11-12 at 10.10.11 PM.png>)

In summary, the models are able to provide insights into the frustrations and pain points faced by Singaporeans, leading them to post toxic and hate filled content onto Reddit. The results particularly reflect a strong undercurrent of racial tension within Singapore and anti-foreigner sentiment in Singapore. 

It is also revealed that there is a pattern behind the topics that cause spikes in toxicity and hate: they often stem from events that occurred in that month or the previous month.

### 4.2 Discussion

In this project, we addressed the business problem of identifying and categorising toxic and hateful content on Singaporean subreddits. The MDDI’s Online Trust and Safety Department can obtain actionable insights into the main topics and themes that cause friction within the online community through usage of this model. The benefit is twofold: the MDDI can use these findings to shape targeted community guidelines, and can shed light on strong but hidden sentiments for the government to act upon. The findings can be used by the MDDI to implement data-driven platform policies aimed at reducing online toxicity. The model’s insights also hold implications for safeguarding vulnerable groups, especially youth and children, by highlighting the prevalence of cyberbullying for instance. The government may also choose to take action to address these prevalent issues raised by the findings, as they may manifest in society; this can be done through campaigns, education or enhanced enforcement. In addition, findings show topics that cause a spike in toxicity commonly stem from controversial events happening in Singapore. Understanding which issues are harmful and predicting spikes in toxicity and hate through analysis of day-to-day events, MDDI can take preemptive actions, such as developing youth-oriented online safety resources or coordinating with educational institutes to increase awareness of potential online hazards regarding these situations. In this sense, it has sufficiently addressed the business problem. 

From a business perspective, sentiment analysis can play a role in identifying shifts in public sentiment within Singaporean subreddits, providing valuable insights for MDDI. This allows MDDI to prioritise and develop targeted interventions in response to rising negativity in the online community, and by understanding shifts in sentiment, together with identifying topics, we can anticipate emerging issues that could drive hate and toxicity. Furthermore, automated sentiment analysis is able to offer substantial savings compared to manual content review, enabling MDDI team to process large volumes of data at scale and identify patterns with reduced time and resources. 

In addition, using BERTopic and zero-shot classification delivers a model that identifies and adapts to dynamic topics efficiently without need for extensive retraining or reclassification. This reduces ongoing costs and resource requirements. We chose a lower threshold for topic classification, accepting some false positives to more vividly capture the general trend of most harmful topics. For the MDDI, this trade-off means that they can rely on the model to be sensitive in classification, such that texts weakly associated to the topic will still be attributed to the topic, understanding that precision may be slightly compromised for clearer distinctions being made.

In terms of interpretability, the Twitter RoBERTa Base Sentiment model outputs intuitive metrics (i.e positive, negative, neutral) that can be easily communicated to non-technical stakeholders. The gradual changes in sentiment over time are directly interpretable, enabling business users to grasp the broader trends without complex data transformations. Additionally, the use of OpenAI’s ChatGPT API was used to assign simple, concise and human-readable representative labels to topic clusters to enhance the model’s interpretability for non-technical users. The prompt was designed to help users have a clearer understanding of how each topic is associated with toxicity. It therefore helps translate abstract cluster data into easily understandable, actionable categories that reflect real concerns in the community.

In terms of fairness, the model uses a custom list of stopwords and keywords specific to the Singaporean context, which may help mitigate bias against colloquial terms that are not harmful but are common in local conversations. However, given the high reliance on RoBERTa Base Sentiment, BERTopic, and Zero-Shot classification, there is a potential for algorithmic bias to be present. These models are trained on predominantly English datasets, lacking representation from a Singapore-specific context, which can lead to misinterpretation. This may result in unique Singaporean expressions, such as Singlish phrases or local slang, being misclassified due to their nuanced meanings not aligning with typical English-language sentiment analysis. Consequently, the model may interpret these expressions differently from their intended meaning, leading to potential misclassification or underrepresentation of certain topics.

Deployability-wise, the model is computationally demanding. Without GPU resources, processing times are considerably increased, limiting scalability. Additionally, usage of OpenAI’s API will also add to a drain on resources. To mitigate these, business users could utilise periodic batch processing to manage the computational load. The model’s flexibility and adaptability in managing new topics without constant retraining still make it a viable long-term solution, balancing effectiveness with resource efficiency.

Overall, this solution addresses the need for efficient identification of toxic and hateful content in the Singaporean context. While there are considerations around resource requirements and fairness, the model provides a strong starting point that can be refined further to address these challenges.


### 4.3 Recommendations

One limitation in the current analysis is the lack of access to original thread texts data. Without this data, our model can only analyze comments, missing context that could clarify the causes and flow of toxic conversations. Including thread data would provide a more comprehensive view, linking specific topics in initial posts to subsequent hate and toxic responses. Additionally, this would allow us to expand sentiment analysis to evaluate whether negative threads correlate with toxic or hateful comments, and enhance topic modeling to assess how certain thread topics incite hate responses. Access to thread-level data and refining our models based on these insights will strengthen MDDI’s ability to address toxic content effectively.
 
Fine-tuning models with embedding output:

To improve the accuracy and contextual sensitivity of our hatefulness and toxicity classification, sentence-level embeddings generated with the SingBERT model can be integrated into the hate and toxicity models. The embeddings generated from SingBERT, which is fine-tuned for Singlish, is well-suited for the Singapore Reddit data and will enhance the models’ ability to detect hate and toxic content. However, fine-tuning models with embedding output can be time consuming and resource-intensive, requiring significant computational resources. Hence, we are unable to proceed with fine-tuning with embedding at the current stage. 


Improving stopwords removal process:

The use of NLTK stopwords library and updating stopwords to include commonly used Singlish terms can be helpful in removing irrelevant words for topic modelling. However, it is not sufficient to fully identify and remove all words that may not be useful for analysis. The process of selecting words that are irrelevant requires a deeper understanding of context and it can be challenging to ensure all non-relevant words for topic modelling have been filtered out. The distinction between useful and irrelevant terms may not always be clearcut, especially when dealing with Singlish in a dataset from Singapore Reddit. If given more time and resources, the use of embedding can be considered to identify irrelevant terms by using clustering or attention mechanisms to identify which words contribute meaningfully to each topic. 

Possible improvements to Topic Modelling Analysis:

We did not have the time to explore certain relationships due to the complexity of the model used. The 3-step process of topic modelling takes a long time, so it was impossible to explore every relationship in depth. For example, the exploration of the differences between content in each subreddit. A cursory pass was made through each subreddit in 2020. The results showed that there were only minor differences in the topics in each subreddit generating toxicity and hatred.
![alt text](<Screenshot 2024-11-12 at 10.11.45 PM.png>)
However, the assessment was made on a yearly level, and some of the monthly topics may have been left out, looking from a year’s perspective. If there was more time and with a better GPU, we could have delved deeper into this, dissecting it to a month by month analysis, where a more granular view may lead to more obvious differences between the topics of each subreddit.

Given more time, analysis could also have been done on the relationship between type of incident and the subreddit is most likely to appear on, so that moderation efforts can be more targeted in the future.

Other relationships that could have been explored, given more time, is the correlation between toxic comments with Reddit moderation. This can analyse the effectiveness of reddit moderation on hiding toxic comments as well as point out any topic areas that reddit moderation misses out on, highlighting a blind spot and danger zone for online moderators from MDDI and Reddit to look at.

Given more time, we could have looked at further improving the tuning of the epsilon hyperparameter in Zero-Shot classification. The highest f1 score returned was not very high, indicating that there is much space for improvement. This is likely due to the poor quality of labelled data used in the training process to obtain the optimal epsilon value. With a larger quantity of well labelled data, the f1 score is likely to increase. With more time, smaller steps could have been used in the epsilon training process as currently, it is in increments of 0.1. These factors will lead to a more optimal value of epsilon being found, which will improve the model’s ability to accurately display the counts for the main topics of the month.


