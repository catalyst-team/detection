ARG CATALYST_VERSION="19.11"

# "-fp16" or ""
ARG CATALYST_WITH_FP16="-fp16"

FROM catalystteam/catalyst:${CATALYST_VERSION}${CATALYST_WITH_FP16}
# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir && rm requirements.txt

CMD mkdir -p /workspace
WORKDIR /workspace
