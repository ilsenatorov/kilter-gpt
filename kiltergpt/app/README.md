# KilterGPT REST API service

---
## Quick start

### Setup environment

To run and operate the application, you have to set the environment variables values.

| Variable | Default value |
|---|---|
| BIND_IP | 0.0.0.0 |
| BIND_PORT | 8000 |
| HOSTNAME | default |
| BACKEND_CORS_ORIGINS | * |
| LOGGING_LEVEL | INFO |
| SUPABASE_URL | http://127.0.0.1:54321 |
| SUPABASE_KEY | XxXxXxXx |

Env vars can be defined using `.env` file.

You can find example in `example.env`.

```bash
cp conf/example.env .env
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For local development you should also install `supabase` to be able to perform storage operations:

For MacOS:
```bash
brew install supabase/tap/supabase
```
### Run project
```bash
uvicorn kiltergpt.app.main:create_app
```

### Local development

Setup `supabase` first:
> All operations below supposed to be performed from the root of this repo.
  `Docker` daemon should also be active.

1. Initialize Supabase to set up the configuration for developing your project locally:
  ```bash
  supabase init
  ```
2. The start command uses Docker to start the Supabase services.
   This command may take a while to run if this is the first time using the CLI.
  ```bash
  supabase start
  ```
Once all of the Supabase services are running, you'll see output containing your local
Supabase credentials. It should look like this, with urls and keys that you'll use in your
`.env` configuration file:
```bash
Started supabase local development setup.

         API URL: http://localhost:54321
          DB URL: postgresql://postgres:postgres@localhost:54322/postgres
      Studio URL: http://localhost:54323
    Inbucket URL: http://localhost:54324
        anon key: eyJh......
service_role key: eyJh......
```

3. Create migration file with kilter tables initialization script:
```bash
supabase migration new kilter_init_scheme
```
This creates a new empty migration: `supabase/migrations/<timestamp>
_kilter_init_scheme.sql.`

You should copy `scripts/sql/init.sql` into that file.

Finally, then we finished with supabase, we can just run:
```bash
python3 __main__.py
```
to start kilter-gpt service

### Run tests
Coming soon ...

## Service structure
```
.
├── README.md
├── __init__.py
├── api
│   ├── __init__.py
│   └── v0    # api version
│       ├── __init__.py
│       └── kilter_gpt.py   # http methods for KilterGPT service
├── config.py
├── db
│   ├── __init__.py
│   ├── client.py   # creates client for supabase operations
│   └── kilter.py   # defines data models used in supabase tables
├── dependencies.py   # dependencies from service&repository layers
├── main.py   # web-app entrypoint
├── models
│   ├── __init__.py
│   └── generation.py   # defines requests/responses schemes for api
├── repository
│   ├── __init__.py
│   └── kilter.py   # operations with storage
└── service
    ├── __init__.py
    └── kilter_gpt.py   # service logic for KilterGPT
```
