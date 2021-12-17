import Vue from "vue";
import Layout from "./Layout.vue";
import App from "./App.vue";
import SymbolsEditor from "./SymbolsEditor.vue";
import OptionAnalysis from "./OptionAnalysis.vue";
import VueRouter from "vue-router";
import VueLazyload from "vue-lazyload";

Vue.use(VueRouter);

Vue.use(VueLazyload, {
    observer: true,
    observerOptions: {
        rootMargin: "400px",
        threshold: 0.1,
    },
});

const routes = [
    { path: "/", component: App },
    { path: "/symbols", component: SymbolsEditor },
    { path: "/options", component: OptionAnalysis },
    { path: "/tags/:tags", component: App },
];

const router = new VueRouter({
    routes,
});

Vue.config.productionTip = false;

new Vue({
    router,
    render: (h) => h(Layout),
}).$mount("#app");
