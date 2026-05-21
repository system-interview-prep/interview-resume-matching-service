# A Comprehensive Study of Resume Ranking

# Techniques and Their Applications

```
Ikjyot Singh
Apex Institute of Technology
Chandigarh University
Mohali, India
23MCC10009@cuchd.in
```
```
Ankit Garg
University Centre for Research and
Development
Chandigarh University
Mohali, India
ankitgitm@gmail.com
```
```
Deepti Sharma
Apex Institute of Technology
Chandigarh University
Mohali, India
deepti.e14308@cumail.in
```
**_Abstract_** **—Automated resume ranking systems have become a
game changer in recruitment by assisting employers in dealing
with large flows of applications and increasing the relevance of
the filter results. Typically, manual culling, sorting, or resume
scanning that searches for relevant keywords can screen for
qualifications and skills but fails to do so in the context of the
position. In this review, an attempt has been made to discuss and
identify advanced resume ranking algorithms, such as machine
learning techniques, and NLP algorithms, such as TF- IDF,
cosine similarity and NER. Such methodologies offer more
effective and accurate results, avoiding simple keyword searches
and understanding the context of a candidate’s experience. These
techniques include source-based filtering, content-based filtering,
and collaborative filtering, and using hybrid solutions has proven
to rank candidates more precisely as the biases are lowered while
fairness shot up. This paper also discusses some of the significant
controversies: the quality of input data, the issues with the
algorithms, and the scalability of the ranking in different sectors.
Besides, it outlines possible future work, including extending
technical AI methods like BERT (Bidirectional Encoder
Representations from Transformers) and the inclusion of the
essential ethical standards that must inform the development of a
fair and transparent hiring platform. Lastly, this paper provides
a comprehensive look into today’s resume ranking processes and
their uses in talent procurement.**

**_Keywords— Resume Ranking, Natural Language Processing
(NLP), Term Frequency-Inverse Document Frequency (TF-IDF),
Cosine Similarity, Named Entity Recognition (NER), Machine
Learning, Applicant Tracking Systems (ATS)_**

I. INTRODUCTION
One key piece of workforce planning is recruitment, which
is how organizations attract and select the right talent. In a
crowded landscape where humans are applying for more jobs
than ever and there are more applications than hiring teams
can handle, hiring managers deserve the best tools to help them
separate the wheat from the chaff early and often [1]. A brief
overview of the background and problem is provided next.

_A. Background_

Recruitment is essential to workforce planning, where
organizations find the best talent to fill existing vacancies.
Earlier, this would require sifting through presumably large
numbers of applicants through their resumes, which can
become tedious as the number of candidates increases. In
general, companies with hundreds of employees get hundreds

```
of applications for a single position, making manual assessment
massive and error-prone. Professional resumes offer an
overview of a candidate’s professional profile, including
achievements, competencies, and experience, and they are the
first screening tools employers use. However, due to the
enhancement of technology, automation of ranking resumes
has become essential to enhance the efficiency of this crucial
process [ 2 ].Resume screening and ranking technologies use
numerous algorithms to screen and sort through resumes and
make the best-match candidate selections. It is crucial for such
systems to use tools such as NLP and Machine learning to
parse resume content better. The traditional manual methods,
which can be somewhat efficient if the scale of applications is
comparatively tiny, prove to be less efficient because of their
inability to be scaled and entail much randomness due to the
decisions made by the recruiters [ 3 ]. As a result, very few
recruitment processes are efficient without incorporating
automated systems that contain data-driven methods.
```
```
B. Problem Statement
Despite the merits of automation, traditional systems of
resume evaluation often imply the use of keyword search as the
only means of analysis, thus missing the point and not being
able to process any context of the candidate’s experience or
education. Such systems care about whether the resume
mentions specific keywords regarding a job description that
does not factor in synonyms or other more general relevant
skills, thus excluding potentially suitable candidates. Also,
since the systems rely on keywords, it is rather challenging to
recognize different forms of presenting resumes and modified
language use, which is unsuitable for different environments
and recruiters’ templates [ 4 ].
```
```
One of the main concerns is bias in algorithms and their
decisions. Machine learning algorithms programmed and
trained on past employment data will replicate those biases,
resulting in prejudiced assessments of candidates from minority
groups[ 5 ]. For the same reason, there are ethical issues with
how these systems compare resumes, as organizations cannot
explain why one resume was ranked higher than another [ 6 ].
```
```
C. Objectives and Scope of the Study
In this review, we provide an integrated view of the history,
success, and shortcomings of the methods used in the ranking
of the resumes with an emphasis on theories and practical’s. It
```

is intended to assess the current methods and discuss the future
trends of automation of the recruitment process.

```
Key Focus Areas:
```
```
 Assess how resumes are ranked, be it simple keyword
matching, more analytical NLP and machine learning
techniques, or something in between.
```
```
 Explore the constraints: for example, data quality,
algorithmic bias, and the ability to adapt to various
sectors.
```
```
 Investigate the potential of new technology like deep
learning and transformer models to enhance contextual
understanding.
```
```
 Explore lifelong learning frameworks to keep systems
in touch with the shifting job landscape.
```
```
 Suggest future research avenues that could help
improve fairness, scalability, and efficiency in
automated recruitment.
```
II. OVERVIEW OF RESUME RANKING TECHNIQUES
Various techniques used in automated resume ranking have,
therefore, progressed, which has brought about increased
precision, speed and equity in the face of the recruitment
undertaking. This section explores the significant paradigms,
starting with the conventional Boolean approach to state-of-
the-art machine learning and mixed strategies that lead to
improved resume scanning. Knowledge of such methodologies
is essential, especially when designing recruitment systems that
can be processed manually and are sustainable in different
circumstances. Also, a summary of this is also provided in
Table I providing information regarding work in these
paradigms by different authors.

_A. Keyword-Based Approaches_

Keyword-based approaches, one of the earliest models of
resume ranking, rely on keyword detection. They identify
specific keywords in resumes that match those used to describe
jobs. While these methods are straightforward, they struggle
with handling different word forms or stems of the same root or
different forms of the same word for different parts of speech,
highlighting their limitations.

Term weight–There are several term weighting techniques,
one of the most often used being TF-IDF (Term Frequency–
Inverse Document Frequency). Among all the keyword-based
techniques, TF-IDF is more complex and widespread. TF-IDF
calculates the ratios of the occurrence of each word in a resume
to the overall occurrence of the same word in the general
resume and job description pool. This approach provides a
better view of resume content than the matching algorithm
since this method understates the usually frequent words and
overstates the words strongly related to the job description.
Nevertheless, the problem of navigating the deeper, contextual
meaning of the terms remains with TF-IDF [ 7 ].

_1) Keyword Matching Techniques:_
Keyword matching, a process of identifying keywords in
resumes that match those in the job listing, has its limitations.
While it enables the use of logical operators like AND, OR,

```
NOT to refine or broaden the search, it struggles to
comprehend the context of terms and the differences in the
ways skills and experiences are stated.
```
```
This approach often leads to the elimination of potentially
worthy candidates who may not use the right keywords, a
drawback that the audience should be aware of [ 8 ].
```
```
B. Machine Learning Techniques
While simple keyword-based methods are used in resume
ranking systems, the emergence of machine learning
techniques has opened new possibilities. Big data approaches,
coupled with machine learning, can significantly enhance
resume evaluation by learning resume patterns and improving
evaluation models. This advancement makes it possible to
consider more candidates, offering a promising future for
resume evaluation.
```
```
1) Supervised Learning Models
Supervised learning models work on tagged data; resumes
are associated with relevance scores, implicitly showing that
the keywords match the job descriptions. The major algorithms
used in these systems are Support Vector Machines (SVMs),
Decision Trees, and Random Forests. For instance, SVMs sort
resumes according to predefined attributes, and they are very
accurate if trained on quality data. On the other hand,
supervised models are susceptible to mimicking the bias of the
training data and consume large amounts of clearly labelled
data in the best-case scenario [ 9 - 10 ].
```
```
2) Unsupervised Learning Models
On the other hand, there are the models which belong to
unsupervised learning and do not use labelled data. However,
rather than analysing the text, they identify features and
categorize resumes by means of comparison. By numerous
measures, the K-Means and Hierarchical Clustering approaches
include grouping candidates with related qualifications and
skill characteristics. These models are useful for large,
disparate data sets, however they may not solve the problem in
the same way as the supervised methods when specific
candidate-job matches exist [ 11 ].
```
```
C. Natural Language Processing (NLP) Approaches
Another interesting direction of the development of resume
ranking systems is based on Natural Language Processing
(NLP), which helped approach the key problem from a new
perspective and interpret the semantic sense of the resumes.
NLP does this by first moving systems past the simple text-
based matching of keywords and instead analysing the content
of resumes in a more natural manner.
```
```
Named Entity Recognition (NER) is an NLP approach that
excels at identifying important data in the resume, such as job
position titles, credentials, qualifications, and skills. This
enhances the recognition of the features of the candidate's
experience and skills. NER is perfect for identifying key
entities such as job titles, qualifications, and certifications in
CVs. For example, it extracts domain-specific credentials (
'Nurse' or 'Public accountant' in healthcare industries) but also
helps to extract unique skills like, 'Kubernetes' for cloud
engineering. By finding equivalent or related named entities,
candidates can be matched accurately using NER. However, it
```

is essential to note that NER can be computationally complex
and may not perform well with non-conventional or irregular
resume formats [1 2 ]. A basic workflow for NER is depicted in
fig 1.

_D. Semantic Similarity Measures_

Cosine and Word Embedding similarity (Word2Vec or
BERT) to measure the relevance of words or phrases in
resumes to the targeted job descriptors regarding the meanings
of the two terms rather than recognizing the appearance of a
similar term. These techniques determine the closeness of
terms into a vector space and provide a much better context for
computing resume contents. Consequently, and for its ability to
capture the essence of language, systems enhanced by BERT
can better gauge the relevance of a candidate's skill set
concerning a set job description [13].
Fig. 1. Framework of the resume screening and ranking system using NE

```
TABLE I. SUMMARY OF LITERATURE REVIEW
```
**References Description Advantages Disadvantages Applications Method/Technique**
[1], [5] Measures word relevance by
frequency in document.

```
Simple and easy to
implement.
```
```
Does not capture the full
context of words ignores
same words
```
```
Used in basic ATS system
to find specific keywords
like ―python‖
```
```
TF-IDF
```
[2], [3] Matches exact keywords from job
descriptions to resumes using
AND, OR etc.

```
Fast and efficient
for small datasets.
```
```
Misses synonyms and
variations of terms.
```
```
Used in early ATS to
quickly filter resumes for
low volume roles
```
```
Keyword Matching
```
[3], [5] Measures similarity between
resume and job description based
on vector space models.

```
Captures
relationships
between terms.
```
```
Still limited to syntactic, not
semantic, similarity.
```
```
Applied in advanced ATS
for text intensive role like
research.
```
```
Cosine Similarity
```
[4], [8] Identifies key entities such as
skills, job titles, and
qualifications.

```
Provides a deeper
understanding of
resume content.
```
```
Computationally expensive;
requires large datasets.
```
```
Used in domain specific
jobs and checking
certificates.
```
```
Named Entity
Recognition (NER)
```
[7], [9] Uses labelled data to predict
resume relevance.

```
High accuracy
when trained on
quality data.
```
```
Prone to overfitting and
biased by training data.
```
```
Used in hiring candidates
based on previous data for
roles
```
```
Supervised Learning
```
[9], [10] Combines multiple techniques
(e.g., TF-IDF, NER, ML).

```
Increased accuracy
and fairness in
rankings.
```
```
More computationally
expensive than simpler
models.
```
```
Used in system for high
volume hiring across
various industries
```
```
Hybrid Models
```
## III. METRICS FOR RESUME RANKING

Given that resume ranking systems are intended to help
improve recruitment, yet they reduce the quality of final
results, it is crucial to assess their performance so that the
systems can deliver results that are as good as they are
efficient. The next part of the paper outlines unanswered
questions and future work: 1) Performance metrics: Details of
measures used to assess the efficacy of the proposed systems
include precision, recall, F1 score, perceived user satisfaction,
and results of real-world trials in sections 4 and 5.

_A. Precision and Recall_

It is well known that precision and recall are two basic
measures employed to assess a system's resume ranking
accuracy.

Accuracy is the number of resumes correctly matched to the
number recommended by the system. It quantifies noise
control, indicating how many resumes the system
recommended as suitable for the particular job are precious for
this position.

Precision in recruitment is of utmost importance as it allows
recruiters to effectively manage the applications received,
discarding those that are irrelevant to the job opening.

## (1)^

```
Recall, on the other hand, is the ratio of resumes flagged by
the system and the number of relevant resumes. There, one can
calculate the ratio of the number of true positives—, or actual
resumes that matched the system criteria—relative to the total
number of resumes resumed and so minimize the group of false
negative resumes, that is, for some
```
## (2)^

```
In recruitment, high recall entails the system retrieving as
many qualified candidates as possible to minimize the loss of
good candidates. Precision and recall scores are both important
while evaluating the resume ranking since they define the
system's success/failure and find the optimal points between
precision and recall.
```
```
B. F1 Score
F1 is the harmony mean of precision and recall, giving the
measurement of both in a single figure. It is more helpful,
especially when the distribution between the relevant and
irrelevant resumes is imbalanced since it prevents focusing on
either precision or recall.
```
```
Input Layer
```
```
Ranking
Algorithm
```
```
Similarity
Calculation
```
```
Output
layer
```
```
Text
Processing
```
```
Named Entity
Recognition
```
```
Feature
Extraction
```

## (3)^

A detailed comparison graph is provided showing the
above-mentioned models compared on the metrics of accuracy,
efficiency and context understanding is provided in fig. 2
depicting the models performance.

Fig. 2. Comparison of Resume Ranking Techniques

_C. Real-World Applications_

Resume ranking accuracy can also be assessed in live HR
technology programs. Resume filtering is more widely
incorporated into ATS, which employers use to process large
numbers of applications. Several real-world applications of
resume ranking systems highlight their benefits:

```
 ATS Integration: Resume ranking has been automated
using NLP and machine learning on various ATS
platforms today. For example, systems that use TF-IDF
and NER can quickly analyse and sort thousands of
resumes, enabling large organizations with many
applications for each vacancy to adopt the process.
```
```
 Bias Mitigation: Some real-world systems, such as tech
giants, explicitly utilize a bias-detection algorithm and
fairness constraints to guarantee minority candidates are
averted by historical data bias [ 14 ].
```
IV. CHALLENGES IN RESUME RANKING
Several challenges affect the performance, bias, and
generalizability of automated resume ranking across industries,
even with advanced AI. These difficulties stem from issues of
data quality, algorithmic bias, and the need for system
flexibility. One crucial aspect that often gets overlooked is the
regularity of updating the model to reflect the current trends in
various fields of employment. This adaptability is key to
ensuring the system remains relevant and effective.

_A. Data Quality and Representation_

Despite its relatively simple formulation, resume ranking
poses several critical questions with susceptibility to low data
quality and misrepresentation. Work resumes are available in
many formats, including semi-structured documents and
unstructured plain text, which hampers system-based extraction
of consistent data. The current formats of writing resumes are
incoherent due to differences in format between or within the
sections that include work experience, qualifications, and

```
hampers the accuracy of the results as mention in Table II.
Moreover, data sparseness is a major hurdle faced by
automated systems as resumes typically contain only partial or
inconsistent information, which is a core input for many of the
desired analyses.
```
```
TABLE II. DATA QUALITY AND REPRESENTATION ISSUES
Challenge Description Impact
Diverse
Resume
Formats
```
```
There are many resume formats,
i.e: PDF, DOCX or scanned
images that make extraction
inconsistent
```
```
Parsing and Data
Extraction issue
affecting rankings
```
```
Unstructured
Data
```
```
Since there is no standard
template to write a resume, it is
written in free form.
```
```
It is hard to extract
features such as
education or work
experience which
affect accuracy.
Inconsistent
Terminology
```
```
Resumes are often widely
different in quoting job titles,
skills, and experiences so each
might all be credited differently
towards the same expertise
domain
```
```
It causes mismatches
in keyword extraction
or feature based
summarization which
results in low
precision
```
```
For example, important information might be missing or
presented in atypical formats or language that automated
systems are likely to misread. These problems reduce the
general accuracy of the ranking algorithms and may result in
poor evaluations of candidates.[ 15 ]. Also, the imbalanced
dataset can be handled using Bi-SMOTE a framework to
handle such uncertainties. [16].
```
```
B. Bias in Algorithms
One of the significant impediments to resume ranking is
that of algorithmic bias. Many automated systems are trained
on past data on hiring, which was possibly riddled with bias as
well. For instance, if an organisation has always preferred
specific demographics, for example, gender, age, race, or
specific education level, the algorithm will replicate these
preferences and disadvantage members of those groups. This
issue has raised the question of whether fairness and equity can
be deconstructed and implemented in automated recruitment.
Reducing algorithmic bias involves developing a well-designed
system with the incorporation of fairness constraints,
periodically auditing the system, and applying mitigation
measures such as re-sampling or discarding over-sensitive
features in the model. Failure to manage bias makes it possible
for automated systems such as resume filters to increase pre-
existing discrimination of minority candidates, especially
women [ 17 ].
```
```
C. Adaptability to Different Industries
Flexibility is another difficulty because resume ranking
systems must be adjusted to conform to various job-search
scenarios. Most industries provide their terminologies,
qualifications, and even skills requirements that separate
industries. For instance, a candidate applying for a software
developer position may have much different qualifications than
one applying to work in the health care field. However, the two
resumes may be run through a similar software. This means
that it may be dangerous to follow the same classical approach
and rankings of candidates without considering the specifics of
different industries. Hence, systems need to be adaptable to
```

cater to differences in the description of jobs and employ sector
requirements while being efficient and precise.

_D. Continuous Learning and Updating_

As the job market continues to open up and expand, so does
the specification that employers demand from prospective
candidates for the jobs they offer them. In other words, if the
resume ranking systems are to remain effective, the resume
ranking paradigms must learn, adapt and evolve as well. Since
the AL algorithms depend on the data and models used, there is
a high chance that the system will become irrelevant.
Therefore, the candidates will not get suitable jobs considering
the existing demand. The second procedural aspect is related to
the retraining of the machine learning models and the necessity
to apply new data reflecting the orientation to the market flow.
Also, feedback received from the recruiters and hiring
managers can be incorporated into the system to improve its
flexibility and incorporate the new changes in the job market.
Even the most sophisticated ranking infrastructures may
become progressively less effective if they are not supported by
ongoing learning and updating [ 18 ].

V. FUTURE DIRECTIONS
That is why several developments are being made that may
enhance the resume ranking system upon its continued usage.
The following section outlines some major future research
areas to address possibilities of new technologies, resume
parsing in an ATS environment, and the need to incorporate
privacy-sensitive and 'fair' techniques.

_A. Emerging Technologies_

The potential for the state-of-art to sort resumes is
immense, based on the meanings of Artificial Intelligence (AI)
and deep learning. Earlier machine learning paradigms,
including supervised and unsupervised learning, are still
practical for use but are restrained by the requirement for
training data labelling and the absence of the ability to analyze
natural languages deeply. Models like BERT (Bidirectional
Encoder Representations from Transformers) enhance resume
ranking by analysing the semantic relationships between terms.
For instance, BERT can discern that 'data scientist' and
'machine learning engineer' have overlapping skill sets, even if
the specific terms do not appear verbatim in a resume along
with some challenges like computation needs and privacy. In
the future, deep learning models that succeed in natural
language understanding tasks could transform the resume
ranking system by making the system review the resume as
naturally as human beings do [1 9 ].

Furthermore, the system's adaptability is a key advantage. It
can benefit from reinforcement learning, allowing it to adapt
online for resume ranking. Imagine a system that learns from
recruiters' feedback and evolves over time, becoming wiser as
it responds to a larger number of resumes and adapts to
changing hiring demands. This adaptability ensures that the
system can create ranks according to the current trends in the
employment market and the preferences of the recruiters,
providing more accurate and suitable selections of candidates.
This adaptability should reassure the audience about the
system's ability to keep up with changing trends.

```
B. Integration with Applicant Tracking Systems (ATS)
Applicant Tracking Systems (ATS) refer to systems utilized
in modern workplace recruitment to handle applications and
employ some techniques for evaluating candidates. However,
with many conventional ATSs, companies depend on simple
keyword matching or Boolean search, which is a problem.
Moreover, the build-in of the advanced concepts of resume
ranking, including Natural Language Processing and machine
learning, into the ATS platform will improve the device's
efficiency.
The future integration will include other features of ATS
systems to expand from keyword matching to using TF-IDF,
cosine similarity, and NER as a rankings filter and also to have
certain course recommendation depending upon domain with
privacy using deep learning[20]. Further, incorporating deep
learning models into ATS will allow for improvements in
semantic analysis so that the recruiters can pick out candidates
who may not necessarily use the right keywords but whom they
need for the position. This innovation will result in a better
shortlisting process, less time spent in the hiring process and a
more effective recruitment process [ 21 - 22 ].We can also have a
multi-criteria recommendation system that can take into
consideration a number of parameters like skills, certification,
experience etc.[2 3 ]
As companies increasingly migrate to cloud-based ATS
systems, the potential for future improvements is vast. These
systems will become more flexible and capable of handling
large quantities of resumes and data sources. The integration
with platforms like LinkedIn and other social sites will make it
easier to access a candidate's work history and skill set,
marking a significant step forward in the industry's progress.
```
```
C. Ethical Considerations
With resume ranking systems getting brighter in the future,
there are some ethical issues, hence the need to consider
privacy and fairness[2 4 ]. This means that some sensitive issues
such as gender, race or age may be unfairly tilted against
women, blacks, or the elderly. This problem needs to be solved
by articulating fairness constraints on the algorithms so that
they do not put off biased data of candidates to arrive at the
ranking. In addition, audits and bias detection should be
conducted periodically to keep track of the system's decisions.
```
```
D. SDN and Load Balancing for Efficient Model
The advanced and sophisticated AI and ML models need
fast and better storage and networks to work efficiently. Using
software defined networking-based data server computing and
improved load management can help solve this for
optimization [2 5 ].
```
```
VI. CONCLUSION AND FUTURE SCOPE
This study reveals how resume ranking has evolved,
simplifying the selection process. Being data-driven, keyword-
based techniques along with their modern machine learning
methods such as NLP can analyse huge amounts of resumes.
Methods like ADV/WebT, TF-IDF, cosine similarity, and NER
enhance capacity to evaluate resumes at a contextual level,
while hybrid models enhance overall performance.
```

Despite these advancements, deployment and accuracy
challenges remain, including those of data uniformity,
algorithmic fairness, and influencing changes in requirements
as the job market continues to evolve.

Further studies need to utilize state-of-the-art AI
algorithms with advanced natural language processing
approaches including deep learning BERT, and various
transformer-based architecture for superior analysis, contextual
exploration, and evaluation. There are also some difficulties
such as computational cost, fairness, or scalability. Phased
rollout and reallocation of cloud resources are the solutions for
these challenges. These next-gen resume ranking systems
would be built to integrate with too many available platforms
including social media to keep having better data exchanges on
real-time to score user-based insights for smarter candidate
profile evaluations and scalability.

```
REFERENCES
```
[1] A. Wang, ―Advancing organizational effectiveness through strategic
workforce planning and technology integration,‖ Advances in
Economics, Management and Political Sciences, vol. 121, no. 1, pp.
107 – 112, Oct. 2024, doi: 10.54254/2754-1169/121/20242362.
[2] S. Maheshwari and M. Rajesh, ―Resume filtering using keyword
matching algorithm,‖ International Journal of Computer Applications,
vol. 183, no. 2, pp. 23–26, 2018.
[3] Z. Tasheva and V. Karpovich, ―Transformation of recruitment process
through implementation of AI solutions,‖ Journal of Management and
Economics, vol. 4, no. 2, pp. 12–17, 2024.
[4] S. Albitar, S. Fournier, and B. Espinasse, ―An effective TF/IDF-based
text-to-text semantic similarity measure for text classification,‖ in Web
Information Systems Engineering–WISE 2014: 15th International
Conference, Thessaloniki, Greece, October 12-14, 2014, Proceedings,
Part I, vol. 15. Springer International Publishing, 2014.
[5] M. Jibril and T. A. Florentina, ―Governing AI in hiring: An effort to
eliminate biased decision,‖ in New Frontiers in Artificial Intelligence.
JSAI-isAI 2024, vol. 14741, T. Suzumura and M. Bono, Eds. Springer,
Singapore, 2024.
[6] M. Raghavan, S. Barocas, J. Kleinberg, and K. Levy, ―Mitigating bias in
algorithmic hiring: Evaluating claims and practices,‖ in Proceedings of
the ACM Conference on Fairness, Accountability, and Transparency
(FAT), pp. 469–481, 2020.
[7] T. Mikolov, K. Chen, G. Corrado, and J. Dean, ―Efficient estimation of
word representations in vector space,‖ arXiv preprint arXiv:1301.3781,
2013.
[8] S. Qaiser and R. Ali, ―Text mining: Use of TF-IDF to examine the
relevance of words to documents,‖ International Journal of Computer
Applications, vol. 181, no. 1, pp. 25–29, 2018.
[9] J. Ramos, ―Using TF-IDF to determine word relevance in document
queries,‖ in Proceedings of the First International Conference on
Machine Learning, 2003.
[10] A. R. Vishaline, R. K. P. Kumar, S. P. VVNS, K. V. K. Vignesh, and P.
Sudheesh, ―An ML-based resume screening and ranking system,‖ in
International Conference on Signal Processing, Computation,
Electronics, Power and Telecommunication (IConSCEPT), pp. 1–6,
IEEE, July 2024.

```
[11] S. B. Kotsiantis, I. Zaharakis, and P. Pintelas, ―Supervised machine
learning: A review of classification techniques,‖ Emerging Artificial
Intelligence Applications in Computer Engineering, vol. 160, no. 1, pp.
3 – 24, 2007.
[12] S. M. Patil and V. D. Shinde, ―Analyzing the effects of K-Means and
Random Forest algorithms on resume classification through
manipulation of cluster sizes and datasets,‖ in 11th International
Conference on Computing for Sustainable Global Development
(INDIACom), pp. 761–766, IEEE, February 2024.
[13] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, ―BERT: Pre-training
of deep bidirectional transformers for language understanding,‖ in
Conference of the North American Chapter of the Association for
Computational Linguistics, pp. 4171–4186, 2019.
[14] B. Green and Y. Chen, ―Disparate interactions: An audit study of bias in
hiring processes,‖ in Proceedings of the 2020 Conference on Fairness,
Accountability, and Transparency, pp. 255–264, 2019.
[15] C. Zhang and H. Wang, ―Resumevis: A visual analytics system to
discover semantic information in semi-structured resume data,‖ ACM
Transactions on Intelligent Systems and Technology (TIST), vol. 10, no.
1, pp. 1–25, 2018.
[16] O. Tigga, J. Pal, and D. Mustafi, "Bi-SMOTE: a novel framework for
handling imbalanced datasets using machine learning techniques
International Journal of Information Technology, vol. 17, pp. 431–445,
```
2025. doi: 10.1007/s41870- 024 - 02224 - y.
[17] S. Kim and J. Lee, ―Addressing bias in AI-driven recruitment: A case
study on algorithmic fairness,‖ Journal of Ethical AI and Recruitment
Practices, vol. 12, no. 3, pp. 102–118, 2024.
[18] C. Shank and M. A. Hitt, ―Real-time learning in artificial intelligence
systems for resume ranking,‖ Journal of Strategic Information Systems,
vol. 26, no. 3, pp. 134–147, 2017.
[19] A. Deshmukh and A. Raut, ―Applying BERT-based NLP for automated
resume screening and candidate ranking,‖ Annals of Data Science, pp.
1 – 13, 2024.
[20] C. S. Kolli, S. Seelamanthula, V. K. Reddy, P. R. Babu, M. R. K. Reddy,
and B. R. Gumpina, "Privacy enhanced course recommendations
through deep learning in Federated Learning environments,"
International Journal of Information Technology ,pp. 1–7, 2024.
[21] G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, and C. Dyer,
―Neural architectures for named entity recognition,‖ in Conference of
the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, pp. 260–270, 2016.
[22] M. F. Ahmad, "Public opinion and persuasion of algorithmic fairness:
assessment of communication protocol performance for use in
simulation-based reinforcement learning training," International Journal
of Information Technology., vol. 16, no. 2, pp. 687-696, 2024.
[23] K. Anwar, A. Zafar, and A. Iqbal, "An efficient approach for improving
the predictive accuracy of multi-criteria recommender system
International Journal of Information Technology., vol. 16, no. 2, pp.
809 – 816, 2024.
[24] O. Galal, A. H. Abdel-Gawad, and M. Farouk, ―Rethinking of BERT
sentence embedding for text classification,‖ Neural Computing and
Applications, vol. 36, no. 32, pp. 20245–20258, 2024.
[25] K. M. Madhura, G. C. Sekhar, A. Sahu, M. P. Karthikeyan, S. Khurana,
M. Shukla, and N. Vashisht, "Software defined network (SDN) based
data server computing system," International Journal of Information
Technology, vol. 17, pp. 607–613, 2025.