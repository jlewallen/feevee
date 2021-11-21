import Vue from "vue";
import App from "./App.vue";
import VueRouter from "vue-router";
import VueLazyload from "vue-lazyload";

Vue.use(VueRouter);

Vue.use(VueLazyload, {
  observer: true,
  observerOptions: {
    rootMargin: "300px",
    threshold: 0.1,
  },
});

const routes = [
  { path: "/", component: App },
  { path: "/:tags", component: App },
];

const router = new VueRouter({
  routes,
});

Vue.config.productionTip = false;

new Vue({
  router,
  render: (h) => h(App),
}).$mount("#app");
