# Data Science Agents — chat UI

React app served by the gateway. Start the product from the repo root
(`make setup`, then `make up`). See the root [README](../README.md).

```bash
npm install
npm run dev      # http://localhost:5173, with the gateway already up
npm run build    # tsc -b && vite build
npm run lint
npm run test
```

Sign-in is the screen `make up` opens. With Keycloak running that is
**admin / admin**. Without Docker, `make up` prints a dev token to paste.
