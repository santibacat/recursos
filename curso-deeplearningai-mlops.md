Curso ML Ops (Machine Learning Engineering for Production) - Deeplearning.ai / Coursera

Putting models into prod is not only plugging the model, but also addressing the model over time with its changes and challenges that come into the workcycle.

# COURSE 1: INTRODUCTION TO MLOPS

## WEEK 1: ML PROJECT LIFECYCLE

Casi todos los proyectos ML tienen un ciclo de producto similar:

1- Scoping: definir el proyecto, que es lo que quieres hacer, cual es X e Y
    * Very important to decide key metrics
    * Estimate resources and timeline
2- Data: definir datos y baseline, etiquetar y organizar los datos
    * First step: is the data labeled consistently? --> las dudas tienen que estar etiquetadas de la misma forma, normalización (en imagen el rango, en sonido el volumen, tiempo de sonido antes-despues de cada sonido...)
3- Modeling: seleccionar el modelo, entrenar y hacer error analysis (e incluso volver al paso 2).
    * ML model consists on code, hyperparameters and data
    * En academia, generalmente el focus es en code + hyperparameters.
    * En cambio, en productos reales, generalmente es mejor dejar el codigo fijo (model), y enfocarse en los hyperparameters y data.
        * No siempre es mejor tener más datos de forma absurda, sino enfocarse en los errores del error analysis.
4- Deployment: poner en produccion, monitorizar y mantener el sistema.
    * Como ya hemos dicho, poner en prod no es solo subir el modelo, sino implementar hardware que hace falta, el frontend, etc
    * Para mantener el modelo generalmente hay que tener en cuenta el data drift (cambio de optica de camaras, cambio de voz de personas que hablan porque lo usan más jovenes, etc)


### MAJOR CHALLENGES

El mayor problema de los modelos entrenados es el **_concept drift / data drift_** que consiste en el cambio de predicciones del modelo cuando los datos de entrada diferen de los originales (por ej: las condiciones de luz son distintas, o ha cambiado algo respecto al training data).

Cuando esto ocurre, queremos saber:

* How has data changed?
    * Gradual change: slow adoption from different people
    * Sudden change: eg during COVID some habits changed rapidly, house pricing crash after Lehman Brothers.

El segundo problema es el **_POC to Production GAP (Proof of Code)_**, que quiere decir que de tener una 'prueba' del codigo de mL funcionando, aun queda mucho trabajo por haceer (el codigo ML solo es un 6-10% de todo el codigo).

Here is a list of checklist of questions you need to answer when facing sofware engineering issues:

* realtime predictions or batch? --> depends if you need fast, realtime
* cloud predictions vs edge/browser? --> depend normally on computation need, internet availability...
* computer resources (CPU/GPU/memory)
* Latency, throughtput(QPS, querys per second)
* Logging --> enough capacity to know why something fails and imrpove the system
* Security and privacy --> eg: medical records may need not to leave organization/edge device

Usually in the first deployment you will face _Software_ issues, and in the mantainance-monitoring deployments you will face _data drift_ issues.


### COMMON DEPLOYMENT CASES

1. NEW PRODUCT / CAPABILITY

2. AUTOMATE/ASSIST WITH MANUAL TASK

3. REPLACE PREVIOUS ML SYSTEM
    * Normally you want gradual ramp up with monitoring (not just replace)
    * Possibility to rollback if you see sth is not working properly

Common __deployment patterns__:

* One common pattern is use _shadow mode_, where ML runs in paralel to human but its output is not used for any decisions
    * It allows to gather data and compare its output to human
* Another pattern (that sometimes is used after the previous) is _canary deployment_, where you roll out to a small fraction initially (~5%)
    * This allows to monitor the system and ramp up traffic gradually
* Another is _blue green deployment_, when you have a router (or a traffic balancer) which sends data to a model (old-blue version), eventually, it switches to send data to the new-green version.
    * The advantage is the easy way to rollback if somethings go wrong.
    * You can do it fast or gradually

Before you know how to deploy it, you must think about the appropriate __degree of automation__:

    * Human only
    * Shadow mode: working but not using the predictions
    * AI assistance
    * Partial automation: if ML prediction it's sent to human; if confident we use ML prediction
        * Normally very good designs
    * Full automation

The 4 firsts are 'human in the loop' deployments because they need human presence.


### MONITORING

The most commmon way to monitor is to use a __Dashboard__ that gives out information (eg: server load, fraction of missing input values...)

* The best approach is think about the things that could go wrong, and then think a few metrics that will detect that problem
* It's ok to use many metrics initially and gradually remove the unused ones
    * Eg: software metrics (memory, compute, latency...)
    * Input metrics (to see if x change): avg input length-volume (for nlp), avg image brightness (img)...
    * Output metrics (y change): # times returns null, # times user redoes search or switch to typing nlp)...
* You may want to set _thresholds_ for alarms (and adapt them over time).

ML modeling is iterative, to monitoring help to improve the deployment of the models.
Usually it's also iterative to choose the right set of metrics to monitor.

When you see that ML model has degrade performance, you need to retrain. _Manual_ retraining is far more common by now than _automatic_ retraining.


### PIPELINE MONITORING

Normally one pipeline involves many ML models, not just one.

* Eg: in NLP, the VAD (voice activation detector) goes before the voice recognition module; if the quality of the VAD gets degraded, the input of the voice recignition changes too.

So we need to come with metrics to monitor the whole pipeline and also the individual components of the pipeline.

You also have to take into accound, __how quickly do data change?__:

* User data = generally slow drift
* Enterprise data (B2B) = can be fast


## WEEK 2: MODELING

Generally when we have to improve models, there are 2 approahes:

* Model-centric AI development: where you focus on improving the model
* Data-centric: where you focus on improve the data, but NOT ALL DATA at the same time. You need techniques to focus on efficiency.

AI system = Code + Data

__CODE__ (algorithm - model): when you have an architecture that performs well, the most important part is to focus on __DATA__. There are some _challenges_ you have to achieve in model devvelopment:

1. Do well on training set (average train set error)

2. Do well on dev/test sets

    * Somethimes this is not enough!! Low average valid error is not enough, because it may not still be successfull for real-world deployment.
        * Disproportionately important examples --> examples that are not common but are very important that we do fine in the real-world (eg: navigational querys in a web search, such as 'google' or 'youtube')
        * Performance on key slices of the dataset --> eg: make sure our models does not discriminate (ej: by gender, ethnics, location... or, in a recommender system, by retailer, product categories...)
        * Rare classes (skew data distribution) --> when we have very little samples of some class, it may be problematic not only for training, but because we may believe the metrics from the test set and they may not be true do to small numper of samples. Eg: hernia in ChexNet.
    * Even for the test set, we can have a model that does not work in real-world (maybe other clinics, other center...). We need to take that into account.

3. DO well on business metrics - project goals


### STEPS TO CREATE A MODEL

#### 1. STABLISH A BASELINE

The first thing we need to do is to stablish a baseline, normally compared to __human-level performance (HLP)__, to know how good machine can do and how much room for improvement we have.

    > Eg: we may see that metrics for a ml baseline for speech recognition in low bandwith are 70% and it seems low, but if human-level performance is 70% too (because bad conditions, doesn't hear well...) maybe we should focus on other task that has more difference between ml and human-level.

* For _unstructured data_ such as images, audio, text... we can use HLP as a baseline.

* For _structured data_ such as tabular data... humans are very bad so we DONT USE HLP as a baseline, we need others.

Ways to stablish a baseline:

* Human-level performance
* Literature search for state-of-the-art / open source
* Quick and dirty implementations (with current models easily implemented)
* If you already have a ML model running, you can use the performance of the older system as baseline.

it indicated what MIGHT be possible to do. And in some cases (HLP) it also give a sense of inductible error-Bayes error.

#### 2. TIPS FOR GETTING STARTED WITH A MODEL

ML is an iterative process (start model + hyperparameters + data > training > error analysis). So, to start:

* __How do I start on modelling__:
    * Seek for literature search (courses, blogs, open-source projects), but normally not better to use the latest fancy model, but one that may work fine and is easily implemented.
    * It's better to have good data (a reasonable algorithm with good data >> great algorithm with bad data)
    
* __Should deployment constraints be taken into accound when picking the model? It depends__:
    * Yes, if the baseline is already stablished and the goal is to deploy (you already know the project is feasable)
    * No, if the purpose is to stablish a baseline or know if a project is possible (you don't know yet if you can make the project work), and it's worth trying.

* __Run sanity-checks for code and algorithm__:
    * Try to overfit a small training dataset before training on a larger model (even better to use just 1 training example: you need to check that you can overfit). That may help you to detect problems with model, preprocessing...


### ERROR ANALYSIS

Normally, at first the ML algorithms not work, so you need to do error analysis to see the best spent of your time to fix it.

The best way to initiate is __error tagging__: you make an spreadsheet with a sample of errors from the test set and make labels of why the cause of the error is (bad labelling, background noise, blurry...). This is an iterative process so you may add  new labels over time-over the analysis and recheck already checked items.

__Useful metrics for each tag:__

* What fraction of error has that tag?
* From that tag, what fraction is missclassified?

In the previous step we saw that we should look at the gap between ML-HLP to prioritize our work. But now, with error analysis, we must take into account other aspects for EACH category:

* How much room for improvement there is
* How frequently that category appears
* How easy is to improve accuracy in that category
* How important is to improve in that category

Because you may see that ML-HLP gap is high, but you don't have much data from that category, so it might be worth to work on something else.

When you __decide a specific category to improve__, you may try some tricks to improve that category:

* Collect more data (so you don't collect all more data but only the categories needed)
* Use data augmentation to get more data
* Improve label accuracy/data quality


One thing difficult to fix are __skewed datasets__, where categories have very imbalanced data (99% no defect-disease, 1% only defect-disease):

* Accuracy is no useful
* You need _confusion matrix_ (actual label vs predicted label), so you can get metrics like _precision_ and _recall_ (along with TN, TP, FN, FP). We combine both metrics in _F1 score_.
    * Precision =  TP / real positives (TP + FP)
    * Recall =  TP / detected positives (TP + FN)
    * F1-score = 2 / (1/P + 1/R)
* Another useful thing from F1 score is that it can be used as _Multi-class metrics_ (you get an evaluation metric for each class and you can also prioritize where to work on).


### PERFORMANCE AUDITING

When we are in the loop (model + hyperparameters + data > training > error analysis), eventually (usually at the end) we come up with auditing performance.

* Brainstorm the ways the system might go wrong
    * Bad performance on subsets of data (ethnics, gender, different devices, prevalence of rude transcriptions...)
    * How common are certain errors (FP; FN...)
    * Performance on rare classes
* Establish metrics to assess performance against these issues (on appropriate slices of data, not all)
    * Eg: mean accuracy for different genders, accents, devices...
    * Eg: check prevalence of offensive words in the output
* Get business/product owner buy-in (to geet help to understand the problem)

After that, you find that you need to fix a specific type of data, so you will have to focus on it performing data iteration (data-centric approach).

### DATA ITERATION



