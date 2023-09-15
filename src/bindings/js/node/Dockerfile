FROM ubuntu:22.04

WORKDIR openvino

RUN apt-get update
RUN apt-get install -y ca-certificates curl gnupg
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

ENV NODE_MAJOR=18
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
RUN apt-get update
RUN apt-get install -y nodejs build-essential

COPY . .

RUN npm i

RUN chmod +x ./ov_runtime/install_dependencies/install_openvino_dependencies.sh
RUN ./ov_runtime/install_dependencies/install_openvino_dependencies.sh -y

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/openvino/ov_runtime/runtime/lib/intel64/"

CMD ["npm", "test"]
