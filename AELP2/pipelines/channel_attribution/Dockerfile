FROM rocker/tidyverse:4.3.1

# Container stub for ChannelAttribution Markov/Shapley job.
RUN R -e "install.packages(c('ChannelAttribution','bigrquery'), repos='https://cloud.r-project.org')" || true

WORKDIR /app
COPY run_ca.R /app/run_ca.R
ENTRYPOINT ["Rscript", "/app/run_ca.R"]

