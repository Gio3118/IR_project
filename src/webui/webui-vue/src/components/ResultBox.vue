<template>
  <EasyDataTable
    :headers="headers"
    :items="rows"
  />
</template>

<script lang="ts">
export default {
    props: {
        result: {
            type: Object,
            required: true
        }
    },
    data() {
        return {
            rows: [{tweet: 'bruh', score: 5, sentiment: 'pos'}, {tweet: 'ayo', score: 4, sentiment: 'pos'}],
            isLoading: false,
            headers: [
                {
                    text: "Tweet",
                    value: "tweet",
                    width: "10%",
                    sortable: true,
                },
                {
                    text: "Score",
                    value: "score",
                    width: "10%",
                    sortable: true,
                },
                {
                    text: "Sentiment",
                    value: "sentiment",
                    width: "10%",
                    sortable: true,
                },
            ],
            sortable: {
                order: "score",
                sort: "asc"
            }
        }
    },
    watch: {
        result: {
            handler: function(newVal) {
                this.rows = newVal.tweets;
            },
            deep: true
        }
    }
    
}
</script>
<style>
    
</style>