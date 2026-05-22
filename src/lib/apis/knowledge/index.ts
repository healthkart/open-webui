import { WEBUI_API_BASE_URL } from '$lib/constants';

export const createNewKnowledge = async (
    token: string,
    name: string,
    description: string,
    accessControl: null | object,
    embed: boolean
) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/create`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			name: name,
			description: description,
			embed: embed,
            access_control: accessControl
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getKnowledgeBases = async (token: string = '', page: number | null = null) => {
	let error = null;
	const searchParams = new URLSearchParams();

	if (page !== null) {
		searchParams.append('page', `${page}`);
	}

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/?${searchParams.toString()}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	if (Array.isArray(res)) {
		if (page === null) {
			return res;
		}

		const pageSize = 50;
		const pageNumber = Math.max(page, 1);
		const start = (pageNumber - 1) * pageSize;
		const end = start + pageSize;
		return {
			items: res.slice(start, end),
			total: res.length
		};
	}

	return res;
};

export const getKnowledgeBaseList = async (token: string = '') => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/list`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const searchKnowledgeBases = async (
	token: string = '',
	query: string | null = null,
	_viewOption: string | null = null,
	page: number | null = null
) => {
	const knowledgeBases = (await getKnowledgeBases(token)) ?? [];
	const normalizedQuery = (query ?? '').trim().toLowerCase();
	let items = knowledgeBases;

	if (normalizedQuery) {
		items = knowledgeBases.filter((knowledgeBase: any) => {
			const name = typeof knowledgeBase?.name === 'string' ? knowledgeBase.name.toLowerCase() : '';
			const description =
				typeof knowledgeBase?.description === 'string' ? knowledgeBase.description.toLowerCase() : '';
			return name.includes(normalizedQuery) || description.includes(normalizedQuery);
		});
	}

	const total = items.length;
	if (page !== null) {
		const pageSize = 50;
		const pageNumber = Math.max(page, 1);
		const start = (pageNumber - 1) * pageSize;
		const end = start + pageSize;
		items = items.slice(start, end);
	}

	return { items, total };
};

export const searchKnowledgeFiles = async (token: string = '', query: string | null = null) => {
	const knowledgeBases = (await getKnowledgeBases(token)) ?? [];
	const normalizedQuery = (query ?? '').trim().toLowerCase();
	const uniqueFiles = new Map<string, any>();

	for (const knowledgeBase of knowledgeBases) {
		const files = Array.isArray(knowledgeBase?.files) ? knowledgeBase.files : [];

		for (const file of files) {
			if (!file?.id) {
				continue;
			}

			if (!uniqueFiles.has(file.id)) {
				uniqueFiles.set(file.id, {
					...file,
					knowledge_id: knowledgeBase?.id
				});
			}
		}
	}

	const files = Array.from(uniqueFiles.values());

	if (!normalizedQuery) {
		return { items: files };
	}

	const items = files.filter((file: any) => {
		const name =
			typeof file?.meta?.name === 'string'
				? file.meta.name.toLowerCase()
				: typeof file?.filename === 'string'
					? file.filename.toLowerCase()
					: '';
		const description = typeof file?.description === 'string' ? file.description.toLowerCase() : '';
		return name.includes(normalizedQuery) || description.includes(normalizedQuery);
	});

	return { items };
};

export const searchKnowledgeFilesById = async (
	token: string = '',
	id: string,
	query: string | null = null,
	_viewOption: string | null = null,
	_permission: string | null = null,
	sortKey: string | null = null,
	page: number | null = null
) => {
	const knowledgeBase = await getKnowledgeById(token, id);
	const normalizedQuery = (query ?? '').trim().toLowerCase();
	const files = Array.isArray(knowledgeBase?.files) ? [...knowledgeBase.files] : [];

	let filtered = files;
	if (normalizedQuery) {
		filtered = files.filter((file: any) => {
			const name =
				typeof file?.meta?.name === 'string'
					? file.meta.name.toLowerCase()
					: typeof file?.filename === 'string'
						? file.filename.toLowerCase()
						: '';
			const description = typeof file?.description === 'string' ? file.description.toLowerCase() : '';
			return name.includes(normalizedQuery) || description.includes(normalizedQuery);
		});
	}

	if (sortKey === 'name') {
		filtered.sort((a: any, b: any) => {
			const aName = (a?.meta?.name ?? a?.filename ?? '').toLowerCase();
			const bName = (b?.meta?.name ?? b?.filename ?? '').toLowerCase();
			return aName.localeCompare(bName);
		});
	}

	const total = filtered.length;
	if (page !== null) {
		const pageSize = 50;
		const pageNumber = Math.max(page, 1);
		const start = (pageNumber - 1) * pageSize;
		const end = start + pageSize;
		filtered = filtered.slice(start, end);
	}

	return { items: filtered, total };
};

export const exportKnowledgeById = async (token: string, id: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${id}/export`, {
		method: 'GET',
		headers: {
			Accept: 'application/octet-stream',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return await res.blob();
		})
		.catch((err) => {
			error = err?.detail ?? 'Failed to export knowledge.';
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getKnowledgeById = async (token: string, id: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${id}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;

			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

type KnowledgeUpdateForm = {
	name?: string;
	description?: string;
	data?: object;
    embed?: boolean,
	access_control?: null | object;
};

export const updateKnowledgeById = async (token: string, id: string, form: KnowledgeUpdateForm) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${id}/update`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			name: form?.name ? form.name : undefined,
			description: form?.description ? form.description : undefined,
			data: form?.data ? form.data : undefined,
            embed: form?.embed,
			access_control: form.access_control
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;

			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const addFileToKnowledgeById = async (token: string, id: string, fileId: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${id}/file/add`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			file_id: fileId
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;

			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const updateFileFromKnowledgeById = async (token: string, id: string, fileId: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${id}/file/update`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			file_id: fileId
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;

			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const removeFileFromKnowledgeById = async (token: string, id: string, fileId: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${id}/file/remove`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			file_id: fileId
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;

			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const resetKnowledgeById = async (token: string, id: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${id}/reset`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;

			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const deleteKnowledgeById = async (token: string, id: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/${id}/delete`, {
		method: 'DELETE',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err.detail;

			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const reindexKnowledgeFiles = async (token: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/knowledge/reindex`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};
