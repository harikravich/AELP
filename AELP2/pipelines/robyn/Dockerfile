FROM rocker/tidyverse:4.3.1

# Container stub for Robyn weekly validator.
# This image is intentionally minimal; in real use, install Robyn and bigrquery.

RUN R -e "install.packages(c('bigrquery'), repos='https://cloud.r-project.org')"
# Optionally: remotes::install_github('facebookexperimental/Robyn')
RUN R -e "install.packages('remotes', repos='https://cloud.r-project.org'); remotes::install_github('facebookexperimental/Robyn')" || true

WORKDIR /app
COPY run_robyn.R /app/run_robyn.R
ENTRYPOINT ["Rscript", "/app/run_robyn.R"]

