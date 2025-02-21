interface SiteConfig {
	site: string
	author: string
	title: string
	shortName: string
	description: string
	lang: string
	ogLocale: string
	shareMessage: string
	paginationSize: number
}

export const siteConfig: SiteConfig = {
	site: 'https://lucas-oma.github.io/', // Write here your website url
	author: 'Lucas O. Martínez', // Site author
	shortName: 'Lucas M.',
	title: 'The Backprop Blog', // Site title.
	description: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.', // Description to display in the meta tags
	lang: 'en-US',
	ogLocale: 'en_US',
	shareMessage: 'Share this post', // Message to share a post on social media
	paginationSize: 6 // Number of posts per page
}
