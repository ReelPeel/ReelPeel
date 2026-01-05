
# 🕵️ Proxy Debugging Cheat Sheet

### 1. Check if it's running
**By Process:**

```bash
ps aux | grep pubmed_proxy.py

```

kill <process number>


**By Port:**

```bash
lsof -i :8080

```

### 2. Watch the Logs

See live traffic and errors:

```bash
tail -f pubmed_proxy.log

```

### 3. Test with Curl

Send a manual request to see if it responds:

```bash
curl "[http://127.0.0.1:8080/proxy/esearch.fcgi?db=pubmed&term=cancer&retmax=1](http://127.0.0.1:8080/proxy/esearch.fcgi?db=pubmed&term=cancer&retmax=1)"

```
