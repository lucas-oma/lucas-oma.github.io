import LinkedInIcon from '@/components/icons/LinkedInIcon'
import GithubIcon from '@/components/icons/GithubIcon'
import EmailIcon from '@/components/icons/EmailIcon'

// ADD YOUR SOCIAL NETWORKS HERE
export const SOCIALNETWORKS = [
	{
		name: 'Github',
		url: 'https://github.com/lucas-oma/',
		icon: GithubIcon
	},

	{
		name: 'LinkedIn',
		url: 'https://www.linkedin.com/in/lucas-martinez-amarilla-924339208/',
		icon: LinkedInIcon
	},

	{
		name: 'Email',
		url: 'mailto:lucasoctaviom@gmail.com',
		icon: EmailIcon
	}
] as const
