# Security policy

## Reporting a vulnerability

Please do not open a public issue. Email [girishgouda6@gmail.com](mailto:girishgouda6@gmail.com).
You can also use **Security → Report a vulnerability** on
[github.com/girishgouda16/data-scientist-agents](https://github.com/girishgouda16/data-scientist-agents)
once private vulnerability reporting is enabled for the repository.

## Before you deploy

- Everything listens on 127.0.0.1 by default. Expose only the gateway (:9001)
  and Keycloak, behind a TLS reverse proxy. Follow [docs/GO_LIVE.md](docs/GO_LIVE.md).
- `make setup` writes random `*_AGENT_API_KEY` values and a random `JWT_SECRET`
  into a new `.env`. Treat every agent key as a full credential: whoever holds
  one can act as any user.
- `docker-compose.yml` ships default passwords marked `CHANGEME`. Change all of
  them before you run it on a shared host.
- Model files (`.pkl`) run code when they load. The agents load only files that
  `export_model` recorded (sha256). Never load a pickle you did not produce.
- SQL and ClickHouse sources run one read-only `SELECT`. Still give the agents a
  database user with read-only rights.
