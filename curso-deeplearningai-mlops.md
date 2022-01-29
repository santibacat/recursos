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

3. DO well on business metrics - project goals



