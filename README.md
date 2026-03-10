# IDAI1041000465-Nihith-Ram-Bikkina

# Building Streamlit Web Apps for visualizing Smart Elevators pathways 

# Candidate Name - Nihith Ram Bikkina

# Candidate Registration Number - 1000465

# CRS Name: Artificial Intelligence

# Course Name - Mathematics for AI

# School name - Birla Open Minds International School, Kollur

# Summative Assessment 

**Project Overview**

This project is a data-driven web application that delivers exploratory data analysis (EDA) and predictive maintenance insights for elevator systems. Built using Streamlit and integrated with Plotly for interactive visualizations, the platform analyzes real elevator sensor data to detect vibration anomalies, identify mechanical wear patterns, and generate structured maintenance recommendations. By analysing inputs such as door revolutions, environmental humidity, pressure (x4), and acceleration (x5), the system provides safe, structured, and accessible engineering insights for building managers and maintenance teams. The application runs on a real dataset of 112,001 sensor readings sampled at 4 Hz during peak evening hours.

**Problem Statement**

Many elevator maintenance programs rely on fixed calendar-based schedules or reactive repair — technicians are dispatched only after a breakdown has already occurred. This leads to costly emergency repairs, extended downtime, and safety risks for building occupants. Without continuous sensor monitoring, early warning signs such as gradually rising vibration levels, dropping pressure readings, and unusual acceleration patterns go undetected until they escalate into critical failures. This project addresses these issues by applying EDA and threshold-based anomaly detection to real sensor data, enabling data-driven, proactive maintenance decisions tailored to actual elevator behaviour.

**Objectives**

* Develop a web-based EDA dashboard for real elevator sensor data using Streamlit and Plotly
* Generate five distinct interactive visualizations covering time series, distributions, scatter relationships, outlier detection, and multivariate correlation
* Detect and quantify anomalous vibration events using an adjustable alert threshold with real-time filtering
* Generate four data-driven maintenance insights grounded in actual Pearson correlation values and quartile-based analysis
* Identify the strongest predictors of vibration across all eight sensor features in a 112,001-row dataset
* Deploy the application on Streamlit Cloud with a public URL and a clean GitHub repository

**Research Summary**

The development process included studying elevator mechanical failure modes, condition-based maintenance techniques, IoT sensor analytics, and existing data-driven monitoring systems. In-depth analysis of the cleaned_missions.csv dataset (112,001 rows, 9 columns) was conducted before development began to identify statistically meaningful patterns. Research on the application of EDA and correlation analysis in predictive maintenance influenced the overall system design. The final application emphasises data accuracy, sensor interpretability, and anomaly transparency, making it suitable for both technical engineers and non-specialist building managers.

**Model Configuration**

Different slider settings were used to optimise the dashboard's analytical performance:
* A default vibration threshold of 70 was selected for anomaly flagging after iterative testing. Thresholds below 50 produced too many false positives, while thresholds above 80 missed meaningful early warning events. At 70, exactly 5,127 readings are flagged (4.6%), aligning with the conventional upper 5th percentile definition of outliers.
* Chart sampling was set to a maximum of 5,000 randomly sampled points after testing confirmed that the full 112,001-row dataset caused significant browser rendering delays. At 5,000 points, charts load in under one second while retaining full visual fidelity. All KPI metrics and correlation values continue to use the full dataset.
* Insights were iteratively refined through multiple rounds of writing and cross-checking — each cited number (correlation value, quartile mean, anomaly count) was verified directly against pandas computations before being included in the final application.

**Sample Outputs and Validation**

The system was tested across the full cleaned_missions.csv dataset to evaluate accuracy, visual clarity, and analytical correctness. Generated outputs were assessed for data fidelity, chart readability, and insight relevance. The vibration time series correctly renders anomaly markers at all threshold-exceeding readings. The correlation heatmap correctly identifies x4 as the strongest predictor (r = -0.141) and humidity as the strongest positive correlator (r = +0.132). Anomaly detection at threshold 70 correctly flags 5,127 readings (4.6%), verified against a direct pandas filter. Box plots correctly separate x4 and x5 onto a second chart due to their significantly larger value scales. Continuous tuning of threshold values, chart sampling sizes, and filter ranges further improved output quality throughout development.

**Web Application Features**

* User input fields for vibration alert threshold, revolution range, and humidity range via interactive sidebar sliders
* File uploader accepting CSV and XLSX formats with full column validation and clear error messages
* Five interactive Plotly charts: vibration time series with anomaly markers, humidity and revolutions histograms, scatter plot with trend line, sensor box plots split by scale, and full correlation heatmap
* KPI row displaying Avg Vibration, Avg Revolutions, Avg Humidity, Peak Vibration, and Anomaly count with percentage
* Four data-driven maintenance insights with real correlation values, quartile group comparisons, and business impact table
* Raw data table with search, anomaly-only filter, threshold row highlighting, and full CSV download
* EDA summary with dataset shape, column descriptions, missing value checks, descriptive statistics, and correlation bar chart
* A clean, light-themed, intuitive interface using Streamlit native components with zero HTML or CSS

**Deployment**

The application was built with Streamlit and connected to the real elevator sensor dataset to generate data-driven outputs. The complete project was hosted on GitHub and deployed on Streamlit Cloud, allowing public access to the web application.

* Create a GitHub repository named: IDAI104-YourStudentID-YourName
* Upload all project files: app.py, elevator_sensor_data.csv, requirements.txt, README.md, and a .streamlit folder containing config.toml
* Log in to share.streamlit.io and click New app
* Connect your GitHub account and select the repository, branch (main), and main file (app.py)
* Click Deploy and wait for the build and dependency installation to complete
* Copy the live public URL and paste it into the README.md under the Live App section
* Test the application to confirm all five charts render correctly and the anomaly count shows 5,127 at threshold 70
* Submit a PDF containing your GitHub repository link and student details to ai.assignments@wacpinternational.org

Live app link: https://coachbot-nihith-ram-bikkina-tysqbrnpj8yu59v96zjkqo.streamlit.app/

