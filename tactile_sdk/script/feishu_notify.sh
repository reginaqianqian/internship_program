
webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/6470126f-9ebd-428b-af42-93da5d753687"
pypi_url="http://10.69.43.36:8081/simple/tactile_sdk"
changelog_url="https://fcn5hvc5qbfs.feishu.cn/docx/GrJ9d2u7Xo1swDxrL3dcLIJjnYg#share-XjfEdsD7WoYH7DxROeGcR9FunJd"
curl -X POST -H "Content-Type: application/json" -d "{
    \"msg_type\":\"post\",
    \"content\":{
        \"post\":{
            \"zh_cn\":{
                \"title\":\"tactile_sdk release\",
                \"content\":[[
                    {\"tag\":\"text\",\"text\":\"version $(cat version)\\n\"},
                    {\"tag\":\"a\",\"text\":\"asset\\n\",\"href\":\"$pypi_url\"},
                    {\"tag\":\"a\",\"text\":\"changelog\",\"href\":\"$changelog_url\"}
                ]]
            }
        }
    }
}" $webhook_url
