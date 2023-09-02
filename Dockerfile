FROM rappdw/docker-java-python:zulu11.43-python3.7.9

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

RUN chmod +x vncorenlp_service.sh

RUN chmod +x vi-ba-nmt-api.sh

EXPOSE 8000

CMD ["/bin/bash", "vi-ba-nmt-api.sh"]

