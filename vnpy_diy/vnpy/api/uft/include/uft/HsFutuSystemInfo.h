#ifndef _HSFUTU_MONITOR_INFO_H
#define _HSFUTU_MONITOR_INFO_H

#ifdef WIN32
#define EXPORT
#else
#define EXPORT  __attribute__((visibility("default")))
#endif

//���ù�Լ
#ifdef _WIN32
#  ifndef HSAPI
#  define HSAPI __stdcall
#  endif
#else
#  define HSAPI 
#endif


#ifdef __cplusplus
extern "C"
{
#endif
/* ����session_id
*@param   in   ��Ҫ���ܵ��ַ���
*@param   inLen ��Ҫ���ܵ��ַ�������
*@param   out   ������ܺ��ָ��,�˻�����������Ҫ���� ((inLen+16)/3+1)*4
*@param	  outlen ���ܺ���ַ�������
*@param   key   ��������Ȩ��
*@return  0: �ɹ� ��0:ʧ��
*/
EXPORT int HSAPI hundsun_encrypt(char* in, int inLen, char* out, int* outLen, char* key);

/*�ɼ�ϵͳ��Ϣ
*@param pszSysInfo:  ���ܺ�Ĳɼ���Ϣ, ��������������492�ֽ�
*@param iSysInfoLen:  ���ܺ�Ĳɼ���Ϣ�ĳ���
*@param pszSysInfoIntegrity:   �ɼ���Ϣ�����ȣ���������������300�ֽ�
*@param iSysInfoIntegrityLen:   �ɼ���Ϣ�����ȵĳ���
*@return 0������1���ն���Ϣ�ɼ�Ϊ��2���ն˲ɼ����ݼ�����Կ�汾�쳣3���ն���Ϣ�����쳣	
*/
EXPORT int HSAPI hundsun_getsysteminfo(char* pszSysInfo, int* iSysInfoLen, char* pszSysInfoIntegrity, int* iSysInfoIntegrityLen);

/*��ȡָ��ɼ���ϸ������Ϣ��ֻ��HUNDSUN_getsysteminfo�ķ���ֵ�쳣��ʶΪ��ʱ����������ȡ���������Ϣ�������塿
*@param pszSysInfoIntegrity:HUNDSUN_getsysteminfo�������صĲɼ���Ϣ������
*@param pszDetailInfo:��ϸ������Ϣ����������������150�ֽ�
*@param iDetailInfoLen:��ϸ������Ϣ����
*@return true����ȡ�ɹ� false����ȡʧ��
*/
EXPORT bool HSAPI hundsun_getdetailerror(char* pszSysInfoIntegrity, char* pszDetailInfo, int* iDetailInfoLen);

/*��ȡ�ɼ���汾��
*@return :�ַ�����ʽ�İ汾��
*/
EXPORT const char* HSAPI hundsun_getversion();
#ifdef __cplusplus
}
#endif

#endif