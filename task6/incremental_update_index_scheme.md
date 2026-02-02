```puml
@startuml
actor "Admin/System" as admin
participant "Local Storage" as docs
participant "Update Script" as script
participant "Embedding Model" as ml
database "FAISS Index" as db
participant "Log File" as logs

admin -> script : Trigger (Cron)
script -> db : Load Existing Index
script -> docs : Scan for all files
docs --> script : Return .txt files
script -> script : Filter NEW files (via Metadata)
alt New files found
    script -> ml : Generate Embeddings (ONLY new)
    ml --> script : Return Vectors
    script -> db : add_documents() & Save
    script -> logs : Write Status & Added Count
else No changes
    script -> logs : Write "No updates"
end
@enduml
```
