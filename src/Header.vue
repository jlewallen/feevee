<template>
    <div class="top">
        <div><router-link to="/">home</router-link></div>
        <div><router-link to="/profile">profile</router-link></div>
        <div><router-link to="/options">options</router-link></div>
        <div>
            links:
            <a href="https://money.cnn.com/data/fear-and-greed/" target="blank">f&g index</a>
        </div>
        <slot></slot>
    </div>
</template>

<script>
export default {
    name: "Header",
    data() {
        return {};
    },
    methods: {},
};
</script>

<style></style>
