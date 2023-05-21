FROM node:16-alpine AS builder
ARG DOMAIN
ENV NODE_ENV=production

WORKDIR /home/node/app
COPY frontend/package.json .
COPY frontend/yarn.lock .
RUN yarn install --frozen-lockfile


COPY frontend/ .
RUN REACT_APP_BACK_URL=${DOMAIN} \
	yarn build


# Bundle static assets with nginx
FROM nginx:alpine

WORKDIR /usr/share/nginx/html

RUN rm -rf ./*
# Copy built assets from buildery
COPY --from=builder /home/node/app/build .
# Add your nginx.conf

RUN rm /etc/nginx/conf.d/default.conf
COPY /frontend/etc/nginx.conf /etc/nginx/conf.d

CMD ["nginx", "-g", "daemon off;"]