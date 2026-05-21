```
Vol. 15, No. 8 , 20 24
```
```
241 | P a g e
```
# Enhanced Resume Screening for Smart Hiring Using

# Sentence-Bidirectional Encoder Representations from

# Transformers (S-BERT)

## Asmita Deshmukh, Anjali Raut

```
Hanuman Vyayam Prasarak Mandal’s College of Engineering and Technology, Amravati, 444605, Maharashtra, India
```
Abstract **—** In a world inundated with resumes, the hiring
process is often challenging, particularly for large organizations.
HR professionals face the daunting task of manually sifting
through numerous applications. This paper presents **‘** Enhanced
Resume Screening for Smart Hiring using Sentence-Bidirectional
Encoder Representations from Transformers (S- **BERT)’ to**
revolutionize this process. For HR professionals dealing with
overwhelming numbers of resumes, the manual screening process
is time consuming and error-prone. To address this, here the
proposed solution is developed for an automated solution
leveraging NLP techniques and a cosine distance matrix. Our
approach involves pre-processing, embed- ding generation using
S-BERT, cosine similarity calculation, and ranking based on
scores. In our evaluation on a dataset of 223 resumes, our
automated screening mechanism demonstrated remarkable
efficiency with a screening speed of 0.233 seconds per resume. The
**system’s accuracy was 90%, showcasing its ability to effectively**
identify relevant resumes. This work presents a powerful tool for
HR professionals, significantly reducing the manual workload and
enhancing the accuracy of identifying suitable candidates. The
societal impact lies in streamlining hiring processes, making them
more efficient and accessible, ultimately contributing to a more
productive and equitable job market.

```
Keywords — S-BERT; resume; automated screening; job; CV
```
I. INTRODUCTION
In the contemporary landscape of recruitment, the influx of
numerous resumes for a single job opening poses a considerable
challenge for Human Resources (HR) professionals [1]. The
traditional method of manual resume screening, while (being)
essential, is not without its shortcomings [2]. This process, laden
with time-consuming intricacies, demands meticulous attention
in detail to ensure the identification of the most qualified
candidates [3]. However, the reliance on conventional keyword
matching methods in automated screening introduces its own set
of challenges, often resulting in false positives and negatives [4].

To address these challenges, this research paper introduces a
pioneering approach to automated resume screening [5].
Leveraging the capabilities of Sentence-Bidirectional Encoder
Representations from Transformers (S-BERT), a cutting-edge
natural language processing (NLP) model, our methodology
offers a novel perspective to the intricate task of identifying the
most suitable candidates for a given role. S-BERT’s unique
ability to generate contextualized representations of text enables
a nuanced understanding of resumes, allowing for the

```
identification of relevant skills and experiences even when not
explicitly articulated.
```
```
A. Bert
BERT, or Bidirectional Encoder Representations from
Transformers, is a groundbreaking language model developed
by Google AI, significantly impacting natural language
processing (NLP). Operating on the Transformer architecture, it
excels in learning intricate relationships between words and
phrases, crucial for understanding textual meaning. BERT
demonstrates state-of-the-art performance across NLP tasks,
including natural language understanding (NLU), natural
language generation (NLG), and natural language inference
(NLI). Its functionality involves tokenizing input text,
embedding tokens into meaningful vectors, adding positional
embeddings, passing tokens through Transformer encoders to
understand relationships, and generating final embeddings for
the desired NLP task. BERT finds applications in enhancing
search engine results, improving machine translation accuracy,
developing context-aware chatbots, and generating concise text
summaries. As an evolving tool, BERT holds immense potential
to transform human-computer interactions.
```
```
B. S-Bert
S-BERT, short for Sentence-BERT, is a BERT model
adaptation designed for sentence embedding computation.
These embeddings, representing sentence meaning, are valuable
for tasks like semantic similarity, clustering, and information
retrieval. In contrast to BERT, which undergoes masked
language modeling and next sentence prediction training, S-
BERT trained on natural language inference. This task involves
predicting if pairs of sentences entail, contradict, or are neutral.
S-BERT utilizes a triplet loss function, minimizing distances for
similar sentence pairs and maximizing them for dissimilar ones.
During application, S-BERT processes each sentence
independently, generating vector outputs for the first token. This
single-pass approach is computationally more efficient than
BERT’s pairwise sentence comparison. S-BERT excels in
producing semantically meaningful embeddings due to its
emphasis on understanding sentence relationships.
Demonstrating superiority over BERT, S-BERT excels in
downstream tasks, including semantic textual similarity,
paraphrase identification, and clustering.
```
```
Fig. 1 shows the process flow of S-BERT (Sentence-
BERT), a machine learning algorithm that uses a shared encoder
and a distance metric to train a model. The shared encoder is
```

```
Vol. 15, No. 8 , 20 24
```
```
242 | P a g e
```
used to train the model, and the distance metric is used to
measure the distance between two points.

```
Fig. 1. Process flow diagram of sentence BERT (S-BERT).
```
```
Process flow:
Input: S-BERT takes two input sentences as input.
```
Shared encoder: The shared encoder is a neural network that
learns to represent each sentence as a dense vector. The shared
encoder is trained using a natural language inference (NLI) task,
where the model is given a pair of sentences and must predict
their relationship: entailment, contradiction, or neutral. This NLI
training enables S-BERT to capture the semantic differences
between sentences, leading to more meaningful sentence
embeddings. The shared encoder generates embeddings for the
two input sentences. The embeddings are dependent on the
inputs, meaning that the embedding for a sentence is different
depending on the other sentence in the pair.

Distance metric: The embeddings are then fed to a distance
metric to calculate the distance between the two sentences. A
common distance metric used for S-BERT is the cosine
similarity.

Output: Based on the distance, a decision is made about
whether the sentences are similar or dissimilar. If the distance is
small, then the sentences are considered to be similar. If the
distance is large, then the sentences are considered to be
dissimilar.

This paper outlines the proposed methodology, which
involves generating embedding from both resumes and job

```
descriptions using S-BERT and subsequently measuring their
alignment through cosine similarity. The ranking of resumes
based on these scores facilitates an efficient and accurate
screening process.
```
```
The effectiveness of our approach is validated through a
comprehensive evaluation on a dataset of 223 resumes, showing
an impressive accuracy of 90%. Beyond these quantitative
metrics, our method’s resilience to common pitfalls such as
keyword stuffing and its efficiency, with a screening speed of
0.233 seconds per resume, mark a significant advancement in
the realm of automated resume screening.
```
```
As the need for streamlined and unbiased hiring processes
intensifies, our research stands as a beacon for HR professionals,
offering a solution that not only enhances efficiency but also
contributes to the broader goals of diversity, equity, and
inclusivity in the workforce. The ensuing sections delve into the
intricacies of our proposed methodology, the experimental
results, and the potential implications of our work on the future
landscape of smart hiring practices.
```
```
Fig. 2 illustrates the conceptual framework of the proposed
Resume Screening System for Smart Hiring using Sentence-
Bidirectional Encoder Representations from Trans- formers (S-
BERT). The model processes 200 resumes in PDF format,
initially converting them to Excel format. Subsequently, text
normalization techniques such as lemmatization and stemming
are applied. Keywords are then extracted, forming sentences for
each resume. S-BERT generates embeddings for these
sentences. A parallel process is conducted on the job description,
creating job description embeddings. Cosine similarity is
computed between the sentence and job description
embeddings, determining the ranking of resumes based on these
similarity scores. While previous studies have made significant
strides in automated resume screening, several gaps remain that
underscore the urgency of our research. First, many existing
systems rely heavily on keyword matching, which can be easily
gamed and may miss candidates with relevant skills expressed
in different terms. Second, the contextual understanding of
resumes has been limited, often failing to capture the nuanced
relationships between skills, experiences, and job requirements.
Third, there has been insufficient focus on mitigating biases in
automated screening processes, potentially perpetuating unfair
hiring practices. To address these gaps, our research proposes
the use of Sentence-BERT (S-BERT), a state-of-the-art natural
language processing model that offers deeper contextual
understanding and semantic analysis of resume content. This
approach allows for more nuanced matching between resumes
and job descriptions, reducing reliance on exact keyword
matches. Additionally, by incorporating cosine similarity
measures, our method provides a more holistic evaluation of
candidate suitability. To tackle bias concerns, proposed solution
suggest rigorous testing and continuous refinement of the model
with diverse datasets. Our research not only aims to enhance the
efficiency of resume screening but also to improve its fairness
and accuracy, thus addressing critical shortcomings in existing
automated hiring systems.
```

```
Vol. 15, No. 8 , 20 24
```
```
243 | P a g e
```
```
Fig. 2. Concept diagram of the proposed resume screening system for smart hiring using Sentence-Bidirectional Encoder Representations from Transformers (S-
BERT).
```
### II. LITERATURE REVIEW

Automated resume screening has evolved as a critical do-
main within the recruitment process, driven by the need for
efficiency, accuracy, and mitigation of biases in hiring [6].
Traditional methods, reliant on manual screening, have proven
to be time-consuming, error-prone, and susceptible to human
biases [7]. This literature review explores key studies and
methodologies in the realm of automated resume screening,
culminating in the proposal of an advanced system utilizing
Sentence-Bidirectional Encoder Representations from Trans-
formers (S-BERT).

Early endeavors in automated resume screening focused on
basic keyword matching and rule-based systems. Kopparapu
introduced a system for information extraction from
unstructured resumes using natural language processing (NLP)
techniques [4]. This marked an initial attempt to streamline the
screening process. However, these early systems struggled with
nuanced contextual understanding.

Recognizing the limitations of keyword-based approaches,
recent years have witnessed a surge in the application of
advanced NLP techniques to resume screening. Gundlapalli et
al. demonstrated the effective use of NLP tools to screen medical
records for evidence of homelessness [8].

Feller et al. [9] explored NLP for predictive modeling of HIV
diagnoses, showcasing the potential for contextual
understanding beyond explicit mentions [ 10 ].

The application of NLP in diverse domains highlights its
versatility. Nayor et al. [11] employed NLP for accurate
calculation of adenoma detection rates in the context of
screening colonoscopies [12]. Trivedi et al. [13] used NLP for
large-scale labeling of clinical records, emphasizing the
potential for automating data extraction from existing records
[1 4 ].

```
The integration of machine learning into resume screening
systems has been pivotal. Roy et al. [15] introduced a machine
learning approach for automating a resume recommendation
system, illustrating the intersection of NLP and machine
learning in enhancing screening mechanisms [1 6 ]. Recent
breakthroughs in NLP, particularly with models like S-BERT,
have brought contextualized representations to the forefront.
Delimayanti et al. [17] presented a content-based suggestion
system using cosine similarity and KNN algorithms [1 8 ]. This
highlighted the importance of contextual understanding, which
is a hallmark of S-BERT. While the literature showcases
promising advancements, challenges remain. Ndukwe et al. [19]
discussed the need for careful development and evaluation of
NLP models to ensure fairness and mitigate biases [ 20 ].
Choi et al. [ 21 ] emphasized the efficiency of resume
screening through NLP but acknowledged the challenges posed
by computational expenses. Recent research on automated
resume screening and ranking has explored various approaches
using natural language processing and deep learning techniques.
Several studies have investigated the use of transformer-based
models like BERT and its variants for this task. James et al. [ 22 ]
and Mukherjee employed DistilBERT and XLM [23] for resume
shortlisting and ranking. Kinger et al. [24] combined YOLOv
for resume parsing with DistilBERT for ranking. Sentence-
BERT (S-BERT), introduced by Reimers and Gurevych [25],
has gained traction for generating semantically meaningful
sentence embeddings. Subsequent work has evaluated and
refined S-BERT, including TA-SBERT. Seo et al., [26] and Chu
et al., [27], these approaches aim to capture nuanced semantic
relationships between resume content and job requirements,
moving beyond simple keyword matching. Additionally,
researchers have explored combining embeddings with other
techniques, such as named entity recognition and domain-
specific knowledge Vanetik and Kogan, [28]; Yu et al., [29], to
further improve matching accuracy [30]. While progress has
been made, challenges remain in mitigating biases and ensuring
fair evaluation across diverse candidate pools [31].
```

```
Vol. 15, No. 8 , 20 24
```
```
244 | P a g e
```
### III. METHODOLOGY

The objective of this study is to develop an automated
resume screening mechanism to assist the Human Resources
(HR) department in the initial filtering of resumes, ensuring the
provision of the most pertinent candidates for further evaluation,
such as interviews. The dataset used in this study consisted of
223 resumes in PDF format. These resumes were collected from
various job applicants across different fields and experience
levels. To facilitate processing, the PDF files were converted to
CSV (Comma-Separated Values) format. This conversion
preserved the textual content of the resumes while organizing it
into a structured tabular format. The CSV structure included
columns for different resume sections such as personal
information, education, work experience, skills, and additional
qualifications. This standardized format allowed for easier
extraction of relevant information and application of natural
language processing techniques. The conversion from PDF to
CSV was performed using a custom Python script that utilized
PDF parsing libraries to extract text and organize it into
appropriate CSV fields. This structured dataset provided a
consistent foundation for the subsequent steps in our automated
resume screening process.

The proposed automated screening mechanism leverages
Natural Language Processing (NLP) techniques and a cosine
distance matrix to evaluate the alignment of resumes with the
corresponding job description. The methodology employs
Sentence-Bidirectional Encoder Representations from Trans-
formers (S-BERT), a sentence-level model, to extract
embeddings that capture the contextual information from the
resumes. These embeddings are then compared to those
generated from the job description, with the aim of ranking the
resumes based on their relevance.

Stop words are common words that do not carry much
meaning and can cause noise in text analysis. Removing stop
words helps to improve the efficiency and accuracy of the NLP
process. Pre-defined list of stop words in English was used to
remove stop words from the resumes and job description.

Lemmatization is a NLP technique that groups words with
the same meaning together. This is done by reducing words to
their root form. For example, the words “cats” and “kittens”
would both be lemmatized to the root word “cat”.
Lemmatization helps to improve the accuracy of the NLP
process by ensuring that words with the same meaning are
treated similarly.

Stemming is a NLP technique that reduces words to their
common stem. This is done by removing suffixes and prefixes.
For example, the words “running” and “ran” would both be
stemmed to the common stem “run”. Stemming when used in
combination with lemmatization, produces better results.

The resumes and job description were preprocessed using
the following steps:

```
 Stop words were removed.
```
```
 Lemmatization was performed.
```
```
 Stemming was performed.
```
```
The S-BERT model was employed to generate embeddings
from the preprocessed text of both resumes and the job de-
scription. Embeddings are numerical representations of words
that capture their meaning and context. S-BERT is a sentence-
level model that generates embeddings that capture the con-
textual information from the sentences.
```
```
Cosine similarity is a metric used to quantify the similarity
between two vectors. In this study, cosine similarity was used to
measure the similarity between the embeddings of the resumes
and the job description. Resumes with higher cosine similarity
scores are considered to be more relevant to the job description.
```
```
The resumes were ranked based on their cosine similarity
scores. The resumes with the highest cosine similarity scores
were ranked at the top of the list.
```
```
This methodology aims to reduce the reliance on subjective
referral-based hiring by introducing an automated screening
process. This approach ensures a more transparent and
standardized selection mechanism, promoting fairness in the
company’s hiring process.
```
```
The keywords from each resume are concatenated to form a
sentence. S-BERT is then applied to these sentences to generate
embeddings of a specific length, e.g., 4x96 bytes.
```
```
Cosine similarity is then used to calculate the matching score
between the job description embedding and each resume
embedding. The matching score is a value between 0 and 1, with
1 being the best match and 0 being no match.
```
```
Finally, the cosine scores are ranked in descending order.
The resume with the highest cosine score is ranked first.
A step-by-step explanation of the process is as given below:
```
```
 Concatenate keywords to form a sentence: The keywords
from each resume are concatenated to form a sentence.
This sentence captures the essence of the resume and
highlights the applicant’s key skills and experience.
```
```
 Generate S-BERT embeddings: S-BERT is applied to the
sentences to generate embeddings of a specific length.
Embeddings are vector representations of text that
capture the semantic meaning and context of the words.
```
```
 Calculate cosine similarity: Cosine similarity is used to
calculate the matching score between the job description
embedding and each resume embedding. Cosine similar-
ity is a measure of similarity between two vectors. The
higher the cosine similarity score, the more similar the
two vectors are.
```
```
 Rank cosine scores: The cosine scores are ranked in
descending order. The resume with the highest cosine
score is ranked first.
```
```
Fig. 3 illustrates the resume matching mechanism utilizing
S-BERT and cosine similarity in our proposed automated
screening system. The process begins with extracting key
sentences from both the resume and the job description. These
sentences are then fed into the S-BERT model, which generates
embeddings - dense vector representations of the text with
dimensions of 4 x 96 bytes for each input. The embeddings
capture the semantic meaning of the sentences, allowing for a
```

```
Vol. 15, No. 8 , 20 24
```
```
245 | P a g e
```
nuanced comparison beyond simple keyword matching. Once
the embeddings are generated, a cosine matching algorithm
computes the similarity between the resume and job description
embeddings. This similarity score quantifies how well the
content of the resume aligns with the requirements outlined in
the job description. Finally, a ranking algorithm uses these
similarity scores to order the resumes, with higher scores
indicating better matches for the position. This approach enables
a more contextual and meaningful comparison between
candidates and job requirements, addressing limitations of
traditional keyword-based screening methods.

```
Fig. 3. Resume matching mechanism using S-BERT and cosine similarity.
```
The proposed resume matching mechanism has a number of
advantages:

```
 It is able to accurately match resumes to job descriptions,
even when the resumes are in different formats.
```
```
 It is able to identify resumes that are relevant to the job
description, even if the resumes do not contain all of the
keywords that are listed in the job description.
```
```
 It is able to rank resumes based on how well they match
the job description, making it easy for recruiters to
identify the most qualified candidates.
```
IV. RESULTS
The automated resume screening mechanism was rigor-
ously (applied) evaluated on a dataset consisting of 223 resumes
in PDF format. The results underscore the efficacy of the
proposed methodology in identifying and ranking relevant
candidates based on alignment with the job description.

1) Screening speed: The screening process demonstrated
re- markable efficiency, achieving a speed of 0.233 seconds per
resume. This rapid processing speed ensures the practical
applicability of the automated mechanism to scenarios
involving substantial resume inflow.
2) Accuracy metrics: The evaluation metrics utilized for
gauging the performance of the screening mechanism
encompassed.
3) Accuracy: The mechanism exhibited an accuracy rate of
90%, indicating a high precision in identifying resumes that
align with job requirements.
4) Precision: Precision, representing the percentage of
correctly identified relevant resumes out of the total identified
as relevant, reached 85%.

```
5) Recall: The recall rate, measuring the percentage of
relevant resumes correctly identified out of the total relevant
resumes, achieved a commendable 75%.
6) Ranking consistency: The ranking mechanism displayed
consistent performance, ensuring that resumes were
consistently and accurately prioritized based on their alignment
with the job description.
7) Efficiency in large-scale processing: Experimental
outcomes suggested that the proposed solution maintains
efficiency even as the dataset scales. This scalability aspect is
crucial for handling real-world scenarios involving a substantial
volume of incoming resumes.
8) Impact on workload: The implementation of the
automated screening mechanism resulted in a substantial
reduction in the workload of the initial screening team. This
points to its potential in enhancing the efficiency of the early
stages of the hiring process.
9) Robustness to updates: The generated embeddings, once
created, demonstrated robustness to updates in resumes. Unless
there were significant changes in the content, the same
embeddings could be reused for subsequent screenings,
contributing to processing efficiency.
```
```
Fig. 4 shows the output of an automated resume screening
system that uses S-BERT to calculate the similarity between
each resume and a job description. The system first extracts
words from the resumes and forms sentences from them. The
top six rectangular brackets contain words extracted from
different resumes and the sentences formed by them in
rectangular brackets. Then, it uses S-BERT to calculate the
similarity between each sentence and the job description. The
second-last line shows the similarity scores between the job
description and each of the 30 resumes, and the last line shows
the execution time in seconds. The system can be used to quickly
identify resumes that are most relevant to a job opening. This
can save recruiters time and help them find the best candidates
for the job.
The graph illustrates the correlation between the number of
resumes and the screening time in an S-BERT-based automated
resume screening system. As the number of resumes increases,
the screening time also rises, but not in a linear fashion. For
instance, screening five resumes takes about 0.3 seconds, while
screening 10 resumes takes approximately 0.9 seconds, and
screening 30 resumes extends to about 4.9 seconds. This non-
linear trend implies that the screening time increases at a varying
rate.
```
```
Several explanations could account for this phenomenon.
One possibility is that the efficiency of the automated screening
algorithm improves with experience, allowing it to swiftly
identify and discard unsuitable resumes by learning patterns.
Conversely, in traditional mechanisms, screeners might
experience fatigue with increased resume volume, resulting in
slower screening times. In summary, the data indicates that the
number of resumes significantly influences screening time. HR
managers adopting this solution should consider this
relationship when planning their workflow.
```

```
Vol. 15, No. 8 , 20 24
```
```
246 | P a g e
```
```
Fig. 4. Output screenshot of the proposed automated resume screening system using S-BERT.
```
Error computation and calculations were conducted based on
feedback from the HR, who serves as the final end user of the
developed system. This proposed work curated seven job
profiles from the HR manager and selected 10 resumes for each
of the seven job descriptions. Some resumes were common,
given that candidates had multiple eligibilities. Throughout the
system, this approach ranked the top three candidates and asked
HR to rank the candidates based on their experience. To assess
the system, the proposed work assigned numerical labels (1 to
10) to all resumes in each case, and HR provided rankings that
precisely matched our numbering system. Table I presents the
results obtained through HR rankings versus the results obtained
with the proposed Automated Screening System (AS).

```
TABLE I. VALIDATION TABLE FOR SEVEN DISTINCT JOB DESCRIPTIONS
(JD) COMPARING RANKINGS FROM HR (HUMAN RESOURCES MANAGER) AND
AS (AUTOMATED SYSTEM)
```
```
Type of Screening Rank 1 Rank 2 Rank 3
HR (JD1) 5 7 8
AS (JD1) 5 8 7
HR (JD2) 6 7 9
AS (JD2) 6 7 9
HR (JD3) 6 3 2
AS (JD3) 3 6 2
HR (JD4) 5 7 3
AS (JD4) 5 7 8
HR (JD5) 4 3 2
AS (JD5) 4 1 7
HR (JD6) 3 4 9
AS (JD6) 3 4 9
HR (JD7) 3 8 9
AS (JD7) 3 8 4
```
```
The error calculations follow the proposed rule base outlined
as follows:
```
```
If a completely new resume appears on the list, not present
in the best three resumes suggested by the HR manager,
proposed solution assigns values based on the rank of the
resume. If the 1st ranked resume is replaced, a value of -1 is
assigned, indicating a 100% error, aligning with our acceptable
3 resumes policy. If a totally new resume appears at the 2nd
rank, the error is set to -0.9, where the minus sign indicates an
issue, and the value between 0 to 1 indicates the extent of the
error in percentage; in this case, 90% unacceptable is denoted by
```
- 0.9. For the 3rd rank, the value is reduced to -0.8 as it is the last
resume on the list that was missed.

```
Our primary objective is to prioritize bringing all 3
recommended resumes to the output and ensuring they are in the
correct order: 1, 2, and 3.
```
```
In another case, when resumes match in the top 3 but are not
in the correct order, the proposed solution provides a table to
validate this scenario. For Case 1, where rank 1 by HR
corresponds to ranks 1, 2, and 3 by AS, the resulting errors are
0, -0.4, and - 0.7. This implies that if the rank 1 resume in HR
matches the rank 1 in AS, there is no error (0). If the rank 1 HR
resume is found at Rank 2 in AS, the error is -0.4, indicating a
40% error. Finally, if the rank 1 HR resume is found at Rank 3
in AS, the error is -0.7.
```
```
Similarly, for rank 2 HR, the values are -0.3 (Rank 1 AS), 0
(Rank 2 AS), and -0.3 (Rank 3 AS). The magnitude of error is
reduced from 0.4 to 0.3 as the position is lowered to rank 2.
```
```
For rank 3 HR, the following errors were computed: -0.4 for
Rank 1 AS, -0.2 for Rank 2 AS, and 0 for Rank 3 as this
approach is utilized for error computation to validate the
effectiveness of the automated screening technique.
```
```
The scatter plot in Fig. 5 illustrates the effectiveness of the
proposed S-BERT-based automated resume screening system
across seven distinct job descriptions: Software Engineer, Data
```

```
Vol. 15, No. 8 , 20 24
```
```
247 | P a g e
```
Scientist, Product Manager, Marketing Manager, Sales
Representative, Customer Support Representative, and Human
Resources Manager. Accuracy is quantified as the percentage of
resumes correctly classified as relevant or not for each job, based
on HR evaluations. The proposed mechanism attains an overall
accuracy of 90% across all job descriptions, with the Sales
Representative role having the lowest accuracy at 43.33%, and
the highest accuracy observed for Data Scientist, Product
Manager, and Customer Support Representative. The overall
error rate of the system is 21.42%, indicating S-BERT’s efficacy
in automating the resume screening process.

```
Fig. 5. Plot of number of resumes vs. time required for screening.
```
This S-BERT-based automated screening tool offers HR
managers a means to efficiently handle large volumes of
resumes while maintaining high accuracy. This efficiency
allows HR managers to redirect their focus toward critical tasks
like candidate interviews and hiring decisions. The presented bar
graph represents the average accuracy of the system. Future
opportunities include testing the mechanism on a larger resume
dataset to affirm accuracy and generalizability, extending its
application to screen for diverse job types, and developing a
user-friendly web or mobile app to enhance accessibility for HR
managers.

V. DISCUSSIONS
The proposed automated resume screening mechanism
presents a paradigm shift in the recruitment landscape, offering
distinct advantages over traditional methods. Firstly, its reliance
on Natural Language Processing (NLP) techniques empowers
the system to extract nuanced information from resumes,
ensuring resilience against tactics like keyword stuffing [10].
This not only enhances the accuracy of candidate evaluation but
also reduces the potential for falsification in the initial screening
stages [4]. Secondly, the incorporation of a cosine distance
matrix for ranking resumes based on alignment with the job
description adds a layer of sophistication [22]. By prioritizing
relevance over specific keywords, the system ensures that the
most suitable candidates rise to the top [18]. This is a crucial
departure from conventional keyword-based screening, aligning
the system with the broader goal of identifying candidates based

```
on their actual qualifications and experience [16]. Despite these
advantages, it’s essential to acknowledge the system’s
developmental stage and the inherent biases that can be present
in NLP models [25]. Rigorous development and evaluation
processes are vital to mitigate biases and ensure fairness.
Moreover, the system’s performance hinges on the quality of
training data, emphasizing the need for diverse and
comprehensive datasets [8] [13]. Importantly, while the
automated screening mechanism streamlines the initial filtering
process, it doesn’t replace human judgment. HR professionals
must review top-ranked resumes for final decisions,
emphasizing the collaborative nature of technology and human
expertise in the hiring process [2].
The proposed automated resume screening mechanism
opens avenues for future research in the realm of human
resources. Firstly, it can be a valuable tool for studying factors
contributing to resume success [19]. Analyzing top-ranked
resumes can provide insights into the skills and experiences
highly valued by employers, guiding both candidates and
educators in aligning with industry expectations [3]. Secondly,
the system acts as a catalyst for developing advanced methods
to enhance the accuracy and fairness of automated resume
screening. Future research could focus on refining NLP
techniques, making them more robust to biases and capable of
capturing richer contextual information from resumes [24].
Thirdly, the proposed mechanism sets the stage for the
development of new tools and resources for HR professionals.
Dashboards visualizing automated screening results could
empower HR teams with quick insights, making the hiring
process more transparent and efficient [20]. The proposed
automated resume screening mechanism not only addresses
current challenges in hiring but also paves the way for future
innovations and research in the dynamic field of human
resources. Future endeavors will likely focus on refining the
system’s performance, addressing identified limitations, and
exploring new frontiers in the evolving intersection of
technology and recruitment practices.
```
```
VI. CONCLUSIONS
Proposed method offers a transformative solution to the
challenges faced in contemporary hiring processes. The
conventional method of manual screening, while crucial, is
marred by inefficiencies, biases, and the overwhelming influx of
resumes. This research introduces an automated screening
mechanism leveraging state-of-the-art natural language
processing (NLP) techniques, particularly S-BERT, to
revolutionize the initial phases of candidate evaluation.
```
```
The results of our evaluation on a dataset of 223 resumes
reveal the remarkable efficiency of the proposed methodology.
With a screening speed of 0.233 seconds per resume, the system
showcased practical applicability in scenarios with substantial
resume inflow. The accuracy metrics demonstrated a high
precision in identifying relevant resumes, with an accuracy of
90%. The ranking mechanism exhibited consistency, ensuring
resumes were prioritized accurately based on their alignment
with job descriptions.
```
```
Beyond quantitative metrics, our automated screening
mechanism significantly reduces the workload on initial
screening teams, presenting a scalable solution for handling
```

```
Vol. 15, No. 8 , 20 24
```
```
248 | P a g e
```
large volumes of incoming resumes. The robustness of
generated embeddings to updates in resumes enhances
processing efficiency, allowing for reuse unless there are
substantial content changes. This work not only contributes to
the efficiency of the hiring process but also aligns with broader
societal goals. By automating and streamlining the screening
process, this manuscript contribute to making hiring practices
more efficient, transparent, and accessible. Moreover, the
adoption of advanced NLP techniques like S-BERT helps
mitigate biases and promotes diversity and inclusivity in
candidate selection.

As one move forward, the implications of this research ex-
tend beyond the immediate context. The automated screening
mechanism presented here not only serves as a tool for HR
professionals but also as a beacon for future developments in
smart hiring practices. The integration of cutting-edge NLP
models signifies a step toward a future where technology
enhances, rather than hinders, the human aspect of hiring.

DECLARATION OF STATEMENTS
Author contribution: The conceptualization was jointly
undertaken by Asmita Deshmukh (AD) and Anjali Raut (AR).
AD was responsible for data collection, coding, and
experimentation. Additionally, AD took the lead in preparing
the initial draft of the manuscript, while AR handled corrections.
Furthermore, data analysis and graphic design were conducted
by AD.

Data and Code availability: Upon a reasonable request, both
the data and code utilized in this research will be provided.

```
Funding: This study is not linked to any funding sources.
```
Conflict of interest and Competing interests: The authors
affirm that they have no conflicts of interest or competing
interests to disclose.

```
REFERENCES
```
[1] V. Sinha and P. Thaly, “A review on changing trend of recruitment
practice to enhance the quality of hiring in global organizations,”
Management: journal of contemporary management issues, vol. 18, no. 2,
pp. 141–156, 2013.
[2] D. S. Chapman and J. Webster, “The use of technologies in the recruiting,
screening, and selection processes for job candidates,” International
journal of selection and assessment, vol. 11, no. 2-3, pp. 113 – 120, 2003.
[3] M. Travis, Mastering the art of recruiting: how to hire the right candidate
for the job. Bloomsbury Publishing USA, 2015.
[4] S. K. Kopparapu, “Automatic extraction of usable information from
unstructured resumes to aid search,” in 2010 IEEE International
Conference on Progress in Informatics and Computing, vol. 1, 2010, pp.
99 – 103.
[5] R. T. Hassan and N. S. Ahmed, “Evaluating of efficacy semantic
similarity methods for comparison of academic thesis and dissertation
texts,” Science Journal of University of Zakho, vol. 11, no. 3, pp. 396–
402, 2023.
[6] E. Derous and A. M. Ryan, “When your resume is (not) turning you down:
Modelling ethnic bias in resume screening,” Human Resource
Management Journal, vol. 29, no. 2, pp. 113–130, 2019.
[7] S. Nabi, “Comparative analysis of AI Vs Human based hiring process: A
Survey,” in 2023 International Conference on Computational Intelligence
and Knowledge Economy (ICCIKE), 2023, pp. 432–437.
[8] A. V. Gundlapalli, M. E. Carter, M. Palmer, T. Ginter, A. Redd, S.
Pickard, S. Shen, B. South, G. Divita, and S. Duvall, “Using natural
language processing on the free text of clinical documents to screen for

```
evidence of homelessness among US veterans,” in AMIA Annual
Symposium Proceedings, vol. 2013, 2013, p. 537.
[9] D. J. Feller, J. Zucker, M. T. Yin, P. Gordon, and N. Elhadad, “Using
clinical notes and natural language processing for automated HIV risk
assessment,” Journal of acquired immune deficiency syndromes (1999),
vol. 77, no. 2, p. 160, 2018.
[10] R. Devika, S. V. Vairavasundaram, C. S. J. Mahenthar, V. Varadarajan,
and K. Kotecha, "A Deep Learning Model Based on BERT and Sentence
Transformer for Semantic Keyphrase Extraction on Big Social Data,"
IEEE Access, vol. 9, pp. 165252–165261, 2021.
[11] J. Nayor, L. F. Borges, S. Goryachev, V. S. Gainer, and J. R. Saltzman,
“Natural language processing accurately calculates adenoma and sessile
serrated polyp detection rates,” Digestive diseases and sciences, vol. 63,
pp. 1794–1800, 2018.
[12] A. Mukherjee, "Resume Ranking and Shortlisting with DistilBERT and
XLM," 2024 IEEE International Conference for Women in Innovation,
Technology & Entrepreneurship (ICWITE), IEEE, pp. 301-304, 2024.
[13] H. M. Trivedi, M. Panahiazar, A. Liang, D. Lituiev, P. Chang, J. H. Sohn,
Y.-Y. Chen, B. L. Franc, B. Joe, and D. Hadley, “Large scale semi-
automated labeling of routine free-text clinical records for deep learning,”
Journal of digital imaging, vol. 32, pp. 30–37, 2019.
[14] B. A. Sherazi, S. Laeer, S. Krutisch, A. Dabidian, S. Schlottau, and E.
Obarcanin, “Functions of mHealth Diabetes Apps That Enable the
Provision of Pharmaceutical Care: Criteria Development and Evaluation
of Popular Apps,” International Journal of Environmental Research and
Public Health, vol. 20, no. 1, p. 64, 2022.
[15] P. K. Roy, S. S. Chowdhary, and R. Bhatia, “A Machine Learning
approach for automation of Resume Recommendation system,” Procedia
Computer Science, vol. 167, pp. 2318–2327, 2020.
[16] Y. S. Swarupa and S. Aruna, "Natural Language Processing for Resume
Screening," zkginternational.com, 2024.
[17] M. K. Delimayanti, M. Laya, B. Warsuta, M. B. Faydhur-rahman, M. A.
Khairuddin, H. Ghoyati, A. Mardiyono, and R. F. Naryanto, “Web-Based
Movie Recommendation System using Content-Based Filtering and KNN
Algorithm,” in 2022 9th International Conference on Information
Technology, Computer, and Electrical Engineering (ICITACEE), 2022,
pp. 314–318.
[18] D. Pawade, T. Joshi, and S. Parkhe, "Survey on Resume and Job Profile
Matching System," 2023 6th International Conference on Computing,
Communication and Automation (ICCCA), IEEE, 2023.
[19] I. G. Ndukwe, C. E. Amadi, L. M. Nkomo, and B. K. Daniel, "Automatic
Grading System Using Sentence-BERT Network," Artificial Intelligence
in Education: 21st International Conference, AIED 2020, Springer, pp.
224 – 227, 2020.
[20] G. M. GR, S. Abhi, and R. Agarwal, "A Hybrid Resume Parser and
Matcher using RegEx and NER," 2023 International Conference on
Computer Communication and Informatics (ICCCI), IEEE, 2023.
[21] H. Choi, J. Kim, S. Joe, and Y. Gwon, "Evaluation of BERT and ALBERT
sentence embedding performance on downstream NLP tasks," 2020 25th
International Conference on Pattern Recognition (ICPR), IEEE, pp. 5482–
5487, 2021.
[22] V. James, A. Kulkarni, and R. Agarwal, "Resume Shortlisting and
Ranking with Transformers," International Conference on Intelligent
Systems and Machine Learning, Springer, pp. 99–108, 2022.
[23] B. Wang and C. C. J. Kuo, "SBERT-WK: A sentence embedding method
by dissecting BERT-based word models," IEEE/ACM Transactions on
Audio, Speech, and Language Processing, vol. 28, pp. 2146– 2157 , 2020.
[24] S. Kinger, D. Kinger, S. Thakkar, and D. Bhake, "Towards smarter hiring:
resume parsing and ranking with YOLOv5 and DistilBERT," Multimedia
Tools and Applications, Springer, 2024.
[25] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings
using siamese BERT-networks," arXiv preprint arXiv:1908.10084, 2019.
[26] J. Seo, S. Lee, L. Liu, and W. Choi, "TA-SBERT: Token Attention
sentence-BERT for improving sentence representation," IEEE Access,
vol. 10, pp. 39119–39128, 2022.
[27] Y. Chu, H. Cao, Y. Diao, and H. Lin, "Refined SBERT: Representing
sentence BERT in manifold space," Neurocomputing, vol. 555, p. 126453,
2023.
```

```
Vol. 15, No. 8 , 20 24
```
```
249 | P a g e
```
[28] Vanetik and G. Kogan, "Job vacancy ranking with sentence embeddings,
keywords, and named entities," Information, vol. 14, no. 8, p. 468, 2023.
[29] S. Yu, J. Su, and D. Luo, "Improving BERT-Based Text Classification
with Auxiliary Sentence and Domain Knowledge," IEEE Access, vol. 7,
pp. 176600–176612, 2019.

```
[30] C. Liu, W. Zhu, X. Zhang, and Q. Zhai, "Sentence Part-Enhanced BERT
with Respect to Downstream Tasks," Complex & Intelligent Systems, vol.
9, no. 1, pp. 463–474, 2023.
[31] M. Alsuhaibani, "Deep Learning-based Sentence Embeddings using
BERT for Textual Entailment," International Journal of Advanced
Computer Science and Applications, vol. 14, no. 8, 2023.
```