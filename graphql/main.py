from fastapi import FastAPI
from ariadne import QueryType,MutationType, make_executable_schema, load_schema_from_path
from ariadne.asgi import GraphQL

type_defs = load_schema_from_path("schema.graphql")

query = QueryType()
mutation = MutationType()

FAKE_DB = {}
NEXT_ID = 1

@query.field("user")
def resolve_user(_, info, id):
    return FAKE_DB.get(id)

@mutation.field("createUser")
def resolve_create_user(_, info, name, email):
    global NEXT_ID
    user = {
        "id": str(NEXT_ID),
        "name": name,
        "email": email
    }

    FAKE_DB[str(NEXT_ID)] = user
    NEXT_ID += 1

    return user

schema = make_executable_schema(type_defs , [query, mutation])

app = FastAPI()
app.mount("/graphql", GraphQL(schema, debug = True))


