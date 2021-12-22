import Vue from "vue";
import Layout from "./Layout.vue";
import Home from "./Home.vue";
import ProfileEditor from "./ProfileEditor.vue";
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
    { path: "/profile", component: ProfileEditor },
    { path: "/options", component: OptionAnalysis },
    { path: "/tags/:tags", component: Home },
    { path: "/", component: Home },
];

const router = new VueRouter({
    routes,
});

Vue.config.productionTip = false;

new Vue({
    router,
    render: (h) => h(Layout),
}).$mount("#app");
