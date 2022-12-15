<template>
  <form
    class="flex justify-center items-center flex-col w-full transition-all space-y-2"
    @submit.prevent="searchEvent"
  >
    <div class="flex space-x-2 w-full justify-center items-center">
      <input
        v-model="search"
        class="border-2 rounded-md px-3 h-10 w-full hover:border-emerald-200 focus:ring focus:ring-emerald-400 focus:outline-none focus:border-emerald-500 transition-all"
        type="text"
        placeholder="Enter a query"
      >
      <button
        class="bg-emerald-500 hover:bg-emerald-400 text-white font-bold py-2 px-4 rounded-md transition-all"
        type="button"
        @click="searchEvent"
      >
        Submit
      </button>
    </div>
    <div class="flex space-x-2 w-full items-center">
      <input
        id="fetch"
        v-model="mode_selector"
        class="accent-emerald-600"
        type="radio"
        name="mode_selection"
        value="fetch"
      >
      <label
        for="fetch"
        class="text-gray-500"
      >Fetch and analyze tweets</label>
      <input
        id="scrape"
        v-model="mode_selector"
        class="accent-emerald-600"
        type="radio"
        name="mode_selection"
        value="scrape"
      >
      <label
        for="scrape"
        class="text-gray-500"
      >Scrape and download tweets</label>
    </div>
  </form>
</template>

<script lang="ts">
export default {
    name: 'SearchBar',
    data() {
        return {
            search: '',
            mode_selector: 'fetch'
        }
    },
    methods: {
        searchEvent() {
            this.$emit('search', {
                query: this.search, 
                selector: this.mode_selector
            });
        }
    }
}
</script>
<style>

</style>