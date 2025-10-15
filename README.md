# Rare Disease Chatbot

## How to Deploy

1. Have a docker
2. Put your API keys in the `.env` file.
3. Run the following command in your terminal:

    ```bash
    docker-compose up -d --build
    ```

4. Open your browser and go to `http://localhost:8501`

5. Enjoy!

## Directory

- `data/`: Your knowledge source for RAG.
- `logs/`: Logs for the application. Dont bother about it.

## Important Notes

1. Make sure to replace the placeholder values in the `.env` file with your actual API keys.
2. The first time you run the application, it may take some time to download the necessary models and dependencies.
3. If you encounter any issues, check the logs in the `logs/` directory for more information.
4. To stop the application, run:

    ```bash
    docker-compose down
    ```

5. To rebuild the application after making changes, run:

    ```bash
    docker-compose up -d --build
    ```

6. Ensure that your `data/` directory contains the necessary documents for the chatbot to function effectively.

## License and Credits

Licensed under [GNU GPLv3](LICENSE).

This works done as part of research in [Universitas Negeri Malang](https://www.um.ac.id/).

Contents on `data/` are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
